"""
Utilities for checkpointing learning dynamics-related states (i.e. activations, weights, grads, etc.)
"""

import os
import re
import copy
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader

from huggingface_hub import upload_folder

# typing imports
import torch.nn as nn
from typing import Dict, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from src.config import CheckpointingConfig
from src.config.checkpointing_config import LearningDynamicsCheckpointingConfig
from lightning.fabric import Fabric


class CheckpointStateExtractor:
    """
    Class to extract and save the states of a model at a given checkpoint step for learning
    dynamics research.
    """

    def __init__(
        self,
        learning_dynamics_config: LearningDynamicsCheckpointingConfig,
        fabric: Fabric,
        model: nn.Module,
    ):
        self.learning_dynamics_config = learning_dynamics_config
        self.fabric = fabric
        self.model = model

    @staticmethod
    def _cleanup_hidden_states(checkpoint_activations, batch_index):
        """
        Cleans up the hidden states if we run out of memory during the forward pass. We want
        to ensure that the hidden states are the same size as the batch index. In practice,
        the activations at a given layer might be more than batch_index because at that layer
        we did not run out of memory (only later).
        """
        for layer_name, activations in checkpoint_activations.items():
            if activations.shape[0] > batch_index:
                checkpoint_activations[layer_name] = activations[:batch_index]

    def extract_states(self, dataloader, compute_gradients: bool = False):
        """
        Perform a forward pass of the model on a given batch of data; assumes that the model
        has hooks setup to save the hidden states at each layer.
        """
        checkpoint_activations = {}
        checkpoint_weights = {}

        forward_hooks = self._setup_forward_hooks(
            checkpoint_activations,
            checkpoint_weights,
        )

        for sub_batch in dataloader:
            _input_ids = torch.tensor(sub_batch["input_ids"], device=self.fabric.device)

            if compute_gradients:
                if "labels" in sub_batch:
                    input_ids = _input_ids
                    labels = torch.tensor(
                        sub_batch["labels"], device=self.fabric.device
                    )
                else:
                    input_ids = _input_ids[:, :-1]
                    labels = _input_ids[:, 1:]
            else:
                input_ids = _input_ids
                labels = None

            if labels is None:
                # we can throw away the outputs, we are only interested in the hidden states
                with torch.no_grad():
                    _ = self.model(input_ids)
            else:
                outputs, _ = self.model(input_ids)
                outputs = outputs.transpose(1, 2)
                loss = F.cross_entropy(outputs, labels)
                self.fabric.backward(loss)

        # cleanup forward hooks
        for hook in forward_hooks:
            hook.remove()

        # -----------------------------------------------------
        # Extract gradients from the target tensors of the model
        # -----------------------------------------------------

        layer_suffixes = self.learning_dynamics_config.layer_suffixes
        checkpoint_gradients = {}
        if compute_gradients:
            for name, param in self.model.named_parameters():
                # only do this for the weight matrix of the layer_suffixes
                if (
                    any(layer_suffix in name for layer_suffix in layer_suffixes)
                    and "weight" in name
                ):
                    assert (
                        param.grad is not None
                    ), "Gradient is None for layer: {name} at step: {step}"
                    name = re.sub(r"\.weight", "", name)
                    checkpoint_gradients[name] = param.grad.detach().cpu()

        # zero out the gradients
        self.model.zero_grad()

        return checkpoint_activations, checkpoint_weights, checkpoint_gradients

    ########################################################
    #
    # Setup forward hooks to save activations and weights at each layer
    #
    ########################################################

    def _setup_forward_hooks(self, checkpoint_activations, checkpoint_weights):
        """
        Setup forward hooks for the model to save activations and weights at each layer.
        """
        forward_hooks = []
        layer_suffixes = self.learning_dynamics_config.layer_suffixes

        for name, module in self.model.named_modules():
            if any(layer_suffix in name for layer_suffix in layer_suffixes):
                _forward_hook = module.register_forward_hook(
                    self._get_forward_hook(
                        name, checkpoint_activations, checkpoint_weights
                    )
                )
                forward_hooks.append(_forward_hook)
        return forward_hooks

    def _get_forward_hook(
        self, module_name, checkpoint_activations, checkpoint_weights
    ):
        def _forward_hook(module, _, module_out):
            sequence_idx = self.learning_dynamics_config.sequence_idx
            activations = module_out[:, sequence_idx, :].detach().cpu()

            # check if there is already a key for the module name
            if module_name not in checkpoint_activations:
                # if there is no key, then we create a new key and store the hidden states
                checkpoint_activations[module_name] = activations

                # extract the weight matrix just once
                weight_matrix = module.weight.detach().cpu()
                checkpoint_weights[module_name] = weight_matrix
            else:
                # if there is already a key, then we concatenate the new hidden states to the existing ones
                checkpoint_activations[module_name] = torch.cat(
                    (checkpoint_activations[module_name], activations)
                )

        return _forward_hook


def compute_learning_dynamics_states(
    checkpointing_config: CheckpointingConfig,
    fabric: Fabric,
    model: nn.Module,
    dataset: Dataset,
    compute_gradients: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute the learning dynamics metrics for a given checkpoint step.
    """

    if fabric.global_rank != 0:
        fabric.barrier()
        return

    # Setting up datalaoder for
    def _collate_fn(batch):
        return {"input_ids": [entry["input_ids"] for entry in batch]}

    sub_batch_size = checkpointing_config.learning_dynamics.sub_batch_size
    dataloader = DataLoader(
        dataset, batch_size=sub_batch_size, shuffle=False, collate_fn=_collate_fn
    )
    extractor_dataloader = fabric.setup_dataloaders(dataloader)

    # creating a copy of model with zero gradients
    _model = copy.deepcopy(model)
    _model.zero_grad()

    # setup forward hooks for the model to save activations and weights at each layer
    state_extractor = CheckpointStateExtractor(
        checkpointing_config.learning_dynamics, fabric, _model
    )
    checkpoint_activations, checkpoint_weights, checkpoint_gradients = (
        state_extractor.extract_states(
            extractor_dataloader, compute_gradients=compute_gradients
        )
    )

    fabric.barrier()

    return {
        "checkpoint_activations": checkpoint_activations,
        "checkpoint_weights": checkpoint_weights,
        "checkpoint_gradients": checkpoint_gradients,
    }


def save_learning_dynamics_states(
    checkpointing_config: CheckpointingConfig,
    fabric: Fabric,
    learning_dynamics_states: Dict[str, torch.Tensor],
    gradient_step: int,
    prefix: str = "train",
    learning_dynamics_dataset: Optional[Dataset] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """
    Save the learning dynamics metrics to the checkpointing directory.

    By default only the learning dynamics states are saved. If the learning dynamics dataset
    is provided, it is also saved; if a tokenizer is provided, the dataset is also detokenized
    (i.e. a new column with the text is added to the dataset).

    The learning dynamics dataset is saved in the checkpointing directory as a HuggingFace
    dataset.

    Creates a versioned checkpoint directory with the following structure:

    {checkpointing_config.runs_dir}/
        └── {checkpointing_config.run_name}/
            └── {checkpointing_config.checkpoints_dir}/
                ├── step_{gradient_step}/
                │   └── {checkpointing_config.learning_dynamics_dir}/ # Learning Dynamics files
                │      ├── {prefix}_activations.pt
                │      ├── {prefix}_weights.pt
                │      └── {prefix}_gradients.pt
                │      └── dataset.hf # if learning_dynamics_dataset is provided
                └── latest -> step_{gradient_step}/

    Args:
        checkpointing_config: The configuration object for checkpointing.
        fabric: The Fabric instance for distributed training.
        learning_dynamics_states: The learning dynamics states to save.
        gradient_step: The gradient step at which the learning dynamics states were computed.
        learning_dynamics_dataset: The dataset containing learning dynamics data,
            including input IDs that need to be decoded.
        tokenizer: The tokenizer used to decode input IDs into text.
    """

    # Only rank 0 process saves checkpoints in distributed training
    if fabric.global_rank != 0:
        fabric.barrier()
        return

    runs_dir = checkpointing_config.runs_dir
    run_name = checkpointing_config.run_name
    checkpoints_dir = checkpointing_config.checkpoints_dir
    learning_dynamics_dir = checkpointing_config.learning_dynamics_dir

    run_path = os.path.join(runs_dir, run_name)
    root_checkpoint_path = os.path.join(run_path, checkpoints_dir)
    checkpoint_path = os.path.join(root_checkpoint_path, f"step_{gradient_step}")
    learning_dynamics_path = os.path.join(checkpoint_path, learning_dynamics_dir)
    os.makedirs(learning_dynamics_path, exist_ok=True)

    # save the learning dynamics states
    for key, value in learning_dynamics_states.items():
        torch.save(value, os.path.join(learning_dynamics_path, f"{prefix}_{key}.pt"))

    if learning_dynamics_dataset is not None:
        if tokenizer is not None:
            # go through dataset and decode the input ids; and add back into dataset
            detokenized_dataset = {"input_ids": [], "text": []}

            for entry in learning_dynamics_dataset:
                input_ids = entry["input_ids"]
                decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                detokenized_dataset["input_ids"].append(input_ids)
                detokenized_dataset["text"].append(decoded_text)

            learning_dynamics_dataset = Dataset.from_dict(detokenized_dataset)

        learning_dynamics_dataset.save_to_disk(learning_dynamics_path)

    if checkpointing_config.save_checkpoint_repo_id is not None:
        # Upload the HF model
        upload_folder(
            folder_path=learning_dynamics_path,
            path_in_repo=learning_dynamics_dir,
            repo_id=checkpointing_config.save_checkpoint_repo_id,
            commit_message=f"Saving Learning Dynamics Datas -- Step {gradient_step}",
            revision=checkpointing_config.run_name,
            token=os.getenv("HF_TOKEN"),
        )

    fabric.barrier()
