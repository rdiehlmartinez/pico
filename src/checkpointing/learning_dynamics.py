"""
Utilities for checkpointing learning dynamics-related states (i.e. activations, weights, grads, etc.)
"""

import torch
import gc
import re
import copy

import torch.nn.functional as F
from torch.utils.data import DataLoader

# typing imports
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict
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

        batch = None

        forward_hooks = self._setup_forward_hooks(
            checkpoint_activations,
            checkpoint_weights,
        )

        if compute_gradients:
            if "labels" in batch:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
            else:
                # NOTE: labels are the input ids shifted over by one
                input_ids = batch["input_ids"][:, :-1]
                labels = batch["input_ids"][:, 1:]
        else:
            input_ids = batch["input_ids"]
            labels = None

        batch_index = 0
        full_batch_size = input_ids.shape[0]

        sub_batch_size = 1
        max_sub_batch_size = None

        # -----------------------------------------------------
        # Extract activations and weights through hooks
        # -----------------------------------------------------

        while batch_index < full_batch_size:
            print(f"batch_index: {batch_index}, full_batch_size: {full_batch_size}")
            try:
                if max_sub_batch_size is not None:
                    sub_batch_size = max_sub_batch_size

                batch_end_index = min(batch_index + sub_batch_size, full_batch_size)
                _inputs = input_ids[batch_index:batch_end_index]

                if labels is not None:
                    # NOTE: If labels are present, then we are iterating over the gradient batches
                    _labels = labels[batch_index:batch_end_index]

                if _labels is None:
                    # we can throw away the outputs, we are only interested in the hidden states
                    with torch.no_grad():
                        _ = self.model(_inputs)
                else:
                    outputs, _ = self.model(_inputs)
                    outputs = outputs.transpose(1, 2)
                    loss = F.cross_entropy(outputs, _labels)
                    self.fabric.backward(loss)

            except RuntimeError as e:
                # NOTE: Exception is thrown when the batch size is too large for the GPU
                print(f"RuntimeError: {e}")

                if sub_batch_size == 1:
                    raise Exception("Batch size of 1 is too large for the GPU")

                sub_batch_size //= 2
                max_sub_batch_size = sub_batch_size

                gc.collect()
                torch.cuda.empty_cache()

                self._cleanup_hidden_states(checkpoint_activations, batch_index)

                continue

            batch_index = batch_end_index

            if max_sub_batch_size is None:
                sub_batch_size *= 2

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


def save_learning_dynamics_metrics():
    """ """
    pass
