import lightning as L
import torch
import torch.nn.functional as F

from model import Pico
from config import PicoConfig

from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader

def main():
    L.seed_everything(42)

    fabric = L.Fabric()
    fabric.launch()

    # Data
    dataset = WikiText2()
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    config = PicoConfig()
    config.tokenizer.vocab_size = dataset.vocab_size

    model = Pico(config, fabric)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    for batch in train_dataloader:
        input_ids, labels = batch
        model_output = model(input_ids).transpose(1, 2)

        print(model_output.shape)
        print(labels.shape)

        # Compute loss
        loss = F.cross_entropy(model_output, labels)

        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        break

if __name__ == "__main__":
    main()