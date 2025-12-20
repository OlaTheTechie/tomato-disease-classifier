"""
module for training the models 
"""

from src.model import SimpleTomatoCNN
from torchmetrics import Accuracy 
import torch
from torch.optim import Adam 
from torch import nn 


device = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer: 
    def __init__(self, model:nn.Module, train_loader, val_loader, device="cpu"): 
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    
    def train_model(self, epochs=1, lr=1e-1): 
        optimizer = Adam(params=self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs): 
            self.model.train()
            total_loss = 0

            for images, labels in self.train_loader: 
                images = images.to(self.device)
                labels = labels.to(device)

                optimizer.zero_grad()
                out = self.model(images)
                loss = criterion(images, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        # log the training loss

        print(f"Epoch: {epoch}/{epochs} | Train Loss: {(total_loss / len(self.train_loader)):4f}")

        # for the validation 
        self.model.eval()
        correct, total = 0, 0 

        with torch.no_grad(): 
            for images, labels in self.val_loader: 
                out = self.model(images)
                preds = out.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Validation Accuracy: {(correct / total):4f}")
