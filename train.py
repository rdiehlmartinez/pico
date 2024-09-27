import click
import lightning as L
import torch.nn.functional as F

from model import Pico

from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader

from utils.initialization import (
    initialize_run_dir, 
    initialize_fabric, 
    initialize_config, 
    initialize_logger,
    initialize_optimizer, 
    initialize_lr_scheduler
)
from utils.checkpointing import (
    load_checkpoint, 
    save_checkpoint,
    save_config
)


@click.command()
@click.option("--model_config_override", type=str, default="") #optional 
@click.option("--training_config_override", type=str, default="") #optional 
@click.option("--evaluation_config_override", type=str, default="") #optional 
def main(model_config_override, training_config_override, evaluation_config_override):
    """
    Core Training Loop. 
    """

    # --- Initial Setup --- #

    # Setup Run Dir
    initialize_run_dir(training_config)

    # Setup Configs; Override with command line args if provided
    model_config = initialize_config(model_config_override, "model")
    training_config = initialize_config(training_config_override, "training")
    evaluation_config = initialize_config(evaluation_config_override, "evaluation")

    # Setup Logger
    logger = initialize_logger(training_config)

    # Setup Fabric
    fabric = initialize_fabric(training_config, logger)

    # Setup Dataset, Tokenizer, and Dataloader
    dataset = WikiText2()
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model_config.tokenizer.vocab_size = dataset.vocab_size

    # Setup Model, Optimizer, and Dataloaders
    model = Pico(model_config, fabric)
    optimizer = initialize_optimizer(model, training_config)
    lr_scheduler = initialize_lr_scheduler(optimizer, training_config)

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # --- Load Checkpoint (if specified) --- #
    if training_config.checkpointing.load_path:
        model, optimizer, lr_scheduler, train_start_step = load_checkpoint(
            fabric, training_config, model, optimizer, lr_scheduler
        )
        fabric.print(f"Loaded checkpoint from {training_config.checkpointing.load_path}")

        # Fast Forward the dataloader to the start step
        for _ in range(train_start_step):
            next(iter(train_dataloader))
    else:
        L.seed_everything(42)
        train_start_step = 0
        fabric.print("Training from scratch!")

        save_config(training_config, model_config, evaluation_config)
        save_checkpoint(fabric, training_config, model, optimizer, lr_scheduler, 0)
        fabric.print("Saved initial model state (step 0)")


    # --- Training Loop --- #

    for batch_idx, batch in enumerate(train_dataloader, start=train_start_step):
        input_ids, labels = batch
        model_output = model(input_ids).transpose(1, 2)

        # Compute loss
        loss = F.cross_entropy(model_output, labels)

        # Maybe log loss
        if batch_idx % training_config.logging.log_every_n_steps == 0:
            fabric.log("loss", loss)
            fabric.print(f"Batch {batch_idx} Loss: {loss.item()}")

        # Maybe save checkpoint 
        if batch_idx % training_config.checkpointing.save_every_n_steps == 0:
            # TODO: Double check that the step_idx is correct 
            save_checkpoint(fabric, training_config, model, optimizer, lr_scheduler, batch_idx)

        # Gradient Step 
        if training_config.optimization.gradient_accumulation_steps % batch_idx == 0:
            fabric.backward(loss / training_config.optimization.gradient_accumulation_steps)
            fabric.clip_gradients(model, optimizer, max_norm=training_config.optimization.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        # --- Break Training Condition --- #
        if batch_idx == training_config.training_steps:
            break

    # --- Final Evaluation --- #
    if training_config.evaluation.eval_every_n_steps:
        fabric.print("Starting Evaluation!")

    fabric.print("Finished Training!")

if __name__ == "__main__":
    main()