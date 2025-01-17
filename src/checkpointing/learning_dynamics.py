"""
Utilities for checkpointing learning dynamics-related states (i.e. activations, weights, grads, etc.)

We save the learning dynamics states in a subdirectory of the checkpointing directory.
"""

import os
import re
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from huggingface_hub import upload_folder

import deepspeed

from src.model import Pico

# typing imports
import torch.nn as nn
from typing import Dict, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from src.config import CheckpointingConfig
from src.config.checkpointing_config import LearningDynamicsCheckpointingConfig
from lightning.fabric import Fabric
from lightning.fabric.strategies import DeepSpeedStrategy


# NOTE: DeepSpeed requires a dummy optimizer to be passed in to the setup function
class DummyOptimizer(optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, defaults={})


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

    def extract_states(self, dataloader, compute_gradients: bool = False):
        """Extracts model states (activations, weights, and optionally gradients).

        Given a dataloader, this function will perform a forward pass of the model on each batch,
        and save the activations and weights at each layer. If compute_gradients is True, it will
        also compute the gradients of the model parameters.

        Args:
            dataloader: The dataloader containing the dataset to extract states from.
            compute_gradients: Whether to compute the gradients of the model parameters.

        Returns:
            A dictionary containing the activations, weights, and optionally gradients of the model.
        """
        checkpoint_activations = {}
        checkpoint_weights = {}

        # NOTE: to extract activations and weights, we need to setup forward hooks on the layers
        # of the model that we are interested in. This is a good intro to forward hooks if you
        # are not familiar: https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
        forward_hooks = self._setup_forward_hooks(
            checkpoint_activations,
            checkpoint_weights,
        )

        ########################################################
        #
        # Forward Pass: Extract activations and weights; and compute gradients
        #
        ########################################################

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
                # NOTE: if we are computing gradients, calling backwards will compute the gradients
                # of the model parameters.
                outputs, _ = self.model(input_ids)
                outputs = outputs.transpose(1, 2)
                loss = F.cross_entropy(outputs, labels)
                self.fabric.backward(loss, model=self.model)

        # cleanup forward hooks - NOTE this is not strictly necessary, since self.model is a
        # deepcopy of the original model; but it is good practice to remove the hooks after the
        # forward pass is complete.
        for hook in forward_hooks:
            hook.remove()

        ########################################################
        #
        # Extract gradients from the target tensors of the model
        #
        ########################################################

        layer_suffixes = self.learning_dynamics_config.layer_suffixes
        checkpoint_gradients = {}
        if compute_gradients:
            for name, param in self.model.named_parameters():
                # only do this for the weight matrix of the layer_suffixes
                if (
                    any(layer_suffix in name for layer_suffix in layer_suffixes)
                    and "weight" in name
                ):
                    if isinstance(self.fabric.strategy, DeepSpeedStrategy):
                        _grad = deepspeed.utils.safe_get_full_grad(param)
                    else:
                        _grad = param.grad

                    assert _grad is not None, f"Gradient is None for layer: {name}"
                    name = re.sub(r"\.weight", "", name)
                    checkpoint_gradients[name] = _grad.detach().cpu()

        # zero out the gradients
        self.model.zero_grad()

        return checkpoint_activations, checkpoint_weights, checkpoint_gradients

    ########################################################
    #
    # Setup forward hooks to save activations and weights at each layer
    #
    ########################################################

    def _setup_forward_hooks(self, checkpoint_activations, checkpoint_weights):
        """Setup forward hooks for the model to save activations and weights at each layer.

        This function will setup forward hooks on the layers of the model that we are interested in.
        The forward hooks will save the activations and weights at each layer whenever the forward pass
        is performed.

        Args:
            checkpoint_activations: A dictionary to store the activations at each layer.
            checkpoint_weights: A dictionary to store the weights at each layer.

        Returns:
            A list of forward hooks. We do this so that we can remove the hooks after the forward pass
            is complete.
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
        """Get a forward hook for a given module.

        This function is called by the _setup_forward_hooks function to setup a forward hook for a given
        module. This functions is a closure that captures the module_name, checkpoint_activations, and
        checkpoint_weights.

        Args:
            module_name: The name of the module to setup a forward hook for.
            checkpoint_activations: A dictionary to store the activations at each layer.
            checkpoint_weights: A dictionary to store the weights at each layer.

        Returns:
            A forward hook for the given module.
        """

        def _forward_hook(module, _, module_out):
            sequence_idx = self.learning_dynamics_config.sequence_idx

            local_activations = module_out[:, sequence_idx, :].detach()

            # Gather activations from all processes using fabric
            gathered_activations = self.fabric.all_gather(local_activations)

            # Reshape from [num_processes, batch_size, hidden_dim] to [total_batch_size, hidden_dim]
            gathered_activations = gathered_activations.reshape(
                -1, gathered_activations.shape[-1]
            )

            # check if there is already a key for the module name
            if module_name not in checkpoint_activations:
                # if there is no key, then we create a new key and store the hidden states
                checkpoint_activations[module_name] = gathered_activations

                # extract the weight matrix just once
                weight_matrix = module.weight.detach().cpu()
                checkpoint_weights[module_name] = weight_matrix
            else:
                # if there is already a key, then we concatenate the new hidden states to the existing ones
                checkpoint_activations[module_name] = torch.cat(
                    (checkpoint_activations[module_name], gathered_activations)
                )

        return _forward_hook


def compute_learning_dynamics_states(
    checkpointing_config: CheckpointingConfig,
    fabric: Fabric,
    model: nn.Module,
    dataset: Dataset,
    compute_gradients: bool = False,
) -> Dict[str, torch.Tensor]:
    """Computes the learning dynamics metrics for a given checkpoint step.

    Uses the CheckpointStateExtractor to extract the activations, weights, and optionally gradients
    of the model at a given checkpoint step.

    Args:
        checkpointing_config: The configuration object for checkpointing.
        fabric: The Fabric instance for distributed training.
        model: The model to extract states from.
        dataset: The dataset to extract states from.
        compute_gradients: Whether to compute the gradients of the model parameters.

    Returns:
        A dictionary containing the activations, weights, and optionally gradients of the model.
    """

    # NOTE: Synchronizing processes for fabric dataloader setup
    fabric.barrier()
    model.to("cpu")  # Offloading model to CPU

    # Setting up Dataloader for learning dynamics
    def _collate_fn(batch):
        return {"input_ids": [entry["input_ids"] for entry in batch]}

    batch_size = checkpointing_config.learning_dynamics.batch_size
    sub_batch_size = batch_size // fabric.world_size
    extractor_dataloader = DataLoader(
        dataset, batch_size=sub_batch_size, shuffle=False, collate_fn=_collate_fn
    )
    extractor_dataloader = fabric.setup_dataloaders(
        extractor_dataloader, use_distributed_sampler=True
    )

    # Create a new model instance with same parameters but zero gradients
    _model = Pico(model.config, fabric=fabric)
    _model.load_state_dict(model.state_dict())

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        _model, _ = fabric.setup(_model, DummyOptimizer(_model.parameters()))
    else:
        _model = fabric.setup(_model)

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

    del _model
    torch.cuda.empty_cache()

    # NOTE: Synchronizing processes for model setup
    fabric.barrier()

    model.to(fabric.device)

    return {
        "checkpoint_activations": checkpoint_activations,
        "checkpoint_weights": checkpoint_weights,
        "checkpoint_gradients": checkpoint_gradients,
    }


def save_learning_dynamics_states(
    checkpointing_config: CheckpointingConfig,
    checkpoint_step: int,
    prefix: str,
    fabric: Fabric,
    learning_dynamics_states: Dict[str, torch.Tensor],
    learning_dynamics_dataset: Optional[Dataset] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> None:
    """Save the learning dynamics metrics to the checkpointing directory.

    By default only the learning dynamics states are saved. If the learning dynamics dataset
    is provided, it is also saved; if a tokenizer is provided, the dataset is also detokenized
    (i.e. a new column with the text is added to the dataset).

    The learning dynamics dataset is saved in the checkpointing directory as a HuggingFace
    dataset.

    Creates a versioned checkpoint directory with the following structure:

    {checkpointing_config.runs_dir}/
        └── {checkpointing_config.run_name}/
            └── {checkpointing_config.checkpoints_dir}/
                ├── step_{checkpoint_step}/
                │   └── {checkpointing_config.learning_dynamics_dir}/ # Learning Dynamics files
                │      ├── {prefix}_activations.pt
                │      ├── {prefix}_weights.pt
                │      └── {prefix}_gradients.pt
                │      └── {prefix}_data/ # if learning_dynamics_dataset is provided
                └── latest -> step_{checkpoint_step}/

    Args:
        checkpointing_config: The configuration object for checkpointing.
        checkpoint_step: The checkpoint step at which the learning dynamics states were computed.
        prefix: The prefix for the learning dynamics states.
        fabric: The Fabric instance for distributed training.
        learning_dynamics_states: The learning dynamics states to save.
        learning_dynamics_dataset: The dataset containing learning dynamics data,
            including input IDs that need to be decoded. (optional)
        tokenizer: The tokenizer used to decode input IDs into text. (optional)
    """

    # Only rank 0 process saves checkpoints in distributed training
    if fabric.global_rank != 0:
        return

    runs_dir = checkpointing_config.runs_dir
    run_name = checkpointing_config.run_name
    checkpoints_dir = checkpointing_config.checkpoints_dir
    learning_dynamics_dir = checkpointing_config.learning_dynamics_dir

    run_path = os.path.join(runs_dir, run_name)
    root_checkpoint_path = os.path.join(run_path, checkpoints_dir)
    checkpoint_path = os.path.join(root_checkpoint_path, f"step_{checkpoint_step}")
    learning_dynamics_path = os.path.join(checkpoint_path, learning_dynamics_dir)
    os.makedirs(learning_dynamics_path, exist_ok=True)

    # save the learning dynamics states
    for key, value in learning_dynamics_states.items():
        if value is not None and len(value) > 0:
            torch.save(
                value, os.path.join(learning_dynamics_path, f"{prefix}_{key}.pt")
            )

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

        learning_dynamics_dataset_path = os.path.join(
            learning_dynamics_path, f"{prefix}_data"
        )
        learning_dynamics_dataset.save_to_disk(learning_dynamics_dataset_path)

    if checkpointing_config.save_checkpoint_repo_id is not None:
        # Upload the HF model
        upload_folder(
            folder_path=learning_dynamics_path,
            path_in_repo=learning_dynamics_dir,
            repo_id=checkpointing_config.save_checkpoint_repo_id,
            commit_message=f"Saving Learning Dynamics Data ({prefix}) -- Step {checkpoint_step}",
            revision=checkpointing_config.run_name,
            token=os.getenv("HF_TOKEN"),
        )
