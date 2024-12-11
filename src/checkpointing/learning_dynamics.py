"""
Utilities for checkpointing learning dynamics-related states (i.e. activations, weights, grads, etc.)
"""

import torch
import gc
import re

import torch.nn.functional as F

# typing imports
import torch.nn as nn
from typing import Dict, Any
from src.config import CheckpointingConfig
from src.config.checkpointing_config import LearningDynamicsCheckpointingConfig
from lightning.fabric import Fabric


class LearningDynamicsStates:
    """
    Class to extract and save the states of a model at a given checkpoint step for learning
    dynamics states.
    """

    def __init__(
        self,
        learning_dynamics_config: LearningDynamicsCheckpointingConfig,
        model: nn.Module,
    ):
        self.learning_dynamics_config = learning_dynamics_config
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

    def extract_states(self, batch, compute_gradients: bool = False):
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

        input_ids = batch["input_ids"]

        if compute_gradients:
            # NOTE: labels are the input ids shifted over by one
            input_ids = input_ids[:, 1:]
            labels = input_ids[:, 1:]
        else:
            labels = None

        batch_index = 0
        full_batch_size = batch.shape[0]

        sub_batch_size = 1
        max_sub_batch_size = None

        # -----------------------------------------------------
        # Extract activations and weights through hooks
        # -----------------------------------------------------

        while batch_index < full_batch_size:
            try:
                if max_sub_batch_size is not None:
                    sub_batch_size = max_sub_batch_size

                batch_end_index = min(batch_index + sub_batch_size, full_batch_size)
                _inputs = batch["input_ids"][batch_index:batch_end_index]

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
                    loss.backward()

            except RuntimeError:
                # NOTE: Exception is thrown when the batch size is too large for the GPU

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

        target_layers_suffix = self.learning_dynamics_config.layer_suffixes
        checkpoint_gradients = {}
        if compute_gradients:
            for name, param in self.model.named_parameters():
                # only do this for the weight matrix of the target_layers_suffix
                if (
                    any(suff_name in name for suff_name in target_layers_suffix)
                    and "weight" in name
                ):
                    assert (
                        param.grad is not None
                    ), "Gradient is None for layer: {name} at step: {step}"
                    name = re.sub(r"\.weight", "", name)
                    checkpoint_gradients[name] = param.grad.detach().cpu()

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
        target_layers = self.learning_dynamics_config.target_layers
        for name, module in self.model.named_modules():
            if any(layer in name for layer in target_layers):
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
            if "attention.query_key_value" in module_name:
                hidden_states_out = (
                    module_out[..., 2 * module_out.shape[-1] // 3 :][:, -1, :]
                    .detach()
                    .cpu()
                )

            elif "attention.dense" in module_name:
                # Get name of the qkv module in the same layer
                qkv_module_name = module._global_module_name.replace(
                    "attention.dense", "attention.query_key_value"
                )
                previous_module_output = checkpoint_activations[qkv_module_name]

                curr_batch_size = module_out.shape[0]
                previous_module_output = previous_module_output[-curr_batch_size:].to(
                    "cuda"
                )

                # NOTE: need to call directly to not activate module hook
                hidden_states_out = F.linear(
                    previous_module_output, module.weight, module.bias
                )
                hidden_states_out = hidden_states_out.detach().cpu()

            elif "mlp.dense_4h_to_h" in module_name:
                hidden_states_out = module_out.detach().cpu()[:, -1, :]

            # check if there is already a key for the module name
            if module_name not in self.checkpoint_activations:
                # if there is no key, then we create a new key and store the hidden states
                checkpoint_activations[module_name] = hidden_states_out

                # extract the weight matrix just once
                weight_matrix = module.weight.detach().cpu()
                checkpoint_weights[module_name] = weight_matrix
            else:
                # if there is already a key, then we concatenate the new hidden states to the existing ones
                checkpoint_activations[module_name] = torch.cat(
                    (checkpoint_activations[module_name], hidden_states_out)
                )

        return _forward_hook


def compute_learning_dynamics_states(
    checkpointing_config: CheckpointingConfig,
    fabric: Fabric,
    model: nn.Module,
    train_data_batch: Dict[str, Any] = None,
    eval_data_batch: Dict[str, Any] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the learning dynamics metrics for a given checkpoint step.
    """

    if fabric.global_rank != 0:
        fabric.barrier()
        return

    # setup forward hooks for the model to save activations and weights at each layer
    checkpoint_states = LearningDynamicsStates(
        checkpointing_config.learning_dynamics, model
    )
    checkpoint_activations, checkpoint_weights, checkpoint_gradients = (
        checkpoint_states.extract_states(train_data_batch, compute_gradients=True)
    )

    # if specified in config, run evaluation on the eval_data_batch
    if eval_data_batch is not None:
        eval_activations, eval_weights, _ = checkpoint_states.extract_states(
            eval_data_batch, compute_gradients=False
        )

    return {
        "checkpoint_activations": checkpoint_activations,
        "checkpoint_weights": checkpoint_weights,
        "checkpoint_gradients": checkpoint_gradients,
        "eval_activations": eval_activations,
        "eval_weights": eval_weights,
    }


def save_learning_dynamics_metrics():
    """ """
    pass
