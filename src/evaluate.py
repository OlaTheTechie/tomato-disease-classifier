from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    f1_score, 
    recall_score, 
    precision_score
)

import torch 
from torch import nn
from torch.utils.data import DataLoader
from src.model import (
    SimplePretrainedTomatoMobileNet, 
    SimpleTomatoCNN, 
    AdvancedTomatoCNN
)

from src.configs import paths
from src.dataset import load_dataset, load_preprocessed, transform

class Evalutator: 
    def __init__(self, model: nn.Module, val_loader: DataLoader): 
        self.model = model
        self.val_loader = val_loader
        self.all_preds = []
        self.all_labels = []

        self.model.eval()
        with torch.no_grad(): 

            for images, labels in self.val_loader:
                preds = self.model(images).argmax(dim=1)
                self.all_labels.append(labels)
                self.all_preds.append(preds)

        self.all_labels = torch.cat(self.all_labels).cpu().numpy()
        self.all_preds = torch.cat(self.all_preds).cpu().numpy()

    def get_f1_score(self): 

        _f1_score = f1_score(self.all_labels, self.all_preds, average="macro")
        return _f1_score 
    
    def get_recall_score(self): 

        return recall_score(self.all_labels, self.all_preds, average="macro")

    def get_confusion_matrix(self): 
        
        return confusion_matrix(self.all_labels, self.all_preds)
    
    def get_precision_score(self): 
        
        return precision_score(self.all_labels, self.all_preds, average="macro")



# testing the evaluator 
# model = SimpleTomatoCNN(n_classes=10)
# # evaluator = Evalutator(model, )



if __name__ == "__main__": 
    train_indices_path = paths.PREPROCESSED_DATA_DIR / "train_indices.json"
    val_indices_path = paths.PREPROCESSED_DATA_DIR / "val_indices.json"
    dataset = load_dataset(folder_path=paths.DATA_DIR, transform=transform)

    loaded_train_set, loaded_val_set = load_preprocessed(
        dataset=dataset, 
        train_index_path=train_indices_path, 
        val_index_path=val_indices_path
        )
    
    # testing the shape of the dataset 
    # image, label = loaded_val_set[0]
    # print(f"image shape: {image.shape} \n target: {label}")

    val_loader = DataLoader(
        loaded_val_set, 
        batch_size=32, 
        shuffle=False
    )
    model = SimpleTomatoCNN(n_classes=10)
    state_dict = torch.load(paths.SAVED_MODELS_PATH / "best-cnn.pth", map_location="cpu")
    model.load_state_dict(state_dict=state_dict)
    
    evaluator = Evalutator(model=model, val_loader=val_loader)

    prec_score = evaluator.get_precision_score()
    f1 = evaluator.get_f1_score()
    conf_matrix = evaluator.get_confusion_matrix()

    print(f"precision score: {prec_score}")
    print(f"f1 score: {f1}")
    print(f"confusion matrix: {conf_matrix}")

    # print(f"images shape: {images.shape} \n labels shape: {labels.shape}")
