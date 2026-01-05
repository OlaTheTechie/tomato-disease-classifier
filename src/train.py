"""
module for training the models 
"""
from tqdm import tqdm, trange
from src.model import SimpleTomatoCNN
from torchmetrics import Accuracy 
import torch
from torch.optim import Adam 
from torch import nn 
from src.configs.paths import SAVED_MODELS_PATH 


# device = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer: 
    def __init__(self, model:nn.Module, train_loader, val_loader, lr=1e-3): 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = Adam(params=self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    
    def train_model(self, epochs=1, lr=1e-3): 
        # optimizer = Adam(params=self.model.parameters(), lr=lr)
        # criterion = nn.CrossEntropyLoss().to(device=self.device)

        best_acc = 0.0

        for epoch in trange(epochs, desc="Epoch Progress"): 
            loop = tqdm(self.train_loader, desc=f"training  epoch {epoch + 1} / {epochs}", leave=False)
            self.model.train()
            total_loss = 0

            for images, labels in loop: 
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(images)
                loss = self.criterion(out, labels)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            # log the training loss
                loop.set_postfix(loss=loss.item())
                
            # print(f"Epoch: {epoch}/{epochs} | Train Loss: {(total_loss / len(self.train_loader)):4f}")

            # validate the performance
            val_acc = self.validate()
            print(f"epoch: {epoch} | val-accuracy: {val_acc}")
            if val_acc > best_acc: 
                print(f"we have a better validation accruacy: {val_acc}")

                torch.save(self.model.state_dict(), SAVED_MODELS_PATH / "best-cnn.pth")

            

    def validate(self): 
        correct, total = 0, 0
        self.model.eval()

        with torch.no_grad(): 
            for images, labels in self.val_loader: 
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        
        validation_accuracy = correct / total
        return validation_accuracy

    def save_model(self, path="simple-tomato-cnn.pth"): 
        save_path = SAVED_MODELS_PATH / path
        torch.save(self.model.state_dict(), save_path)
        print("model saved successfully!")

