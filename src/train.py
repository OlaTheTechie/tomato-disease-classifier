"""
module for training the models 
"""

from src.model import SimpleTomatoCNN
from torchmetrics import Accuracy 
import torch
from torch.optim import Adam 
from torch import nn 
from src.configs.paths import SAVED_MODELS_PATH 


# device = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer: 
    def __init__(self, model:nn.Module, train_loader, val_loader): 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

    
    def train_model(self, epochs=1, lr=1e-1): 
        optimizer = Adam(params=self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device=self.device)

        for epoch in range(epochs): 
            self.model.train()
            total_loss = 0

            for images, labels in self.train_loader: 
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                out = self.model(images)
                loss = criterion(out, labels)

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
                images, labels = images.to(self.device), labels.to(self.device)
                out = self.model(images)
                preds = out.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Validation Accuracy: {(correct / total):4f}")

    def save_model(self, path="simple-tomato-cnn.pth"): 
        save_path = SAVED_MODELS_PATH / path
        torch.save(self.model.state_dict(), save_path)
        print("model saved successfully!")

