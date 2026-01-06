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
    def __init__(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor): 
        self.model = model
        self.images = images 
        self.labels = labels 
        self.preds = self.model(self.images).argmax(dim=1)

    def get_f1_score(self): 
        _f1_score = f1_score(self.labels, self.preds)
        return _f1_score 
    
    def get_recall_score(self): 
        return recall_score(self.labels, self.preds)

    def get_confusion_matrix(self): 
        return confusion_matrix(self.labels, self.preds)
    
    def get_precision_score(self, true_val, pred_val): 
        return precision_score(true_val, pred_val)



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

    images = torch.stack([loaded_val_set[i][0] for i in range(len(loaded_val_set))])
    labels = torch.tensor([loaded_val_set[i][1] for i in range(len(loaded_val_set))])

    model = SimpleTomatoCNN(n_classes=10)
    model.eval()
    evaluator = Evalutator(model=model, images=images, labels=labels)

    prec_score = evaluator.get_precision_score()
    f1 = evaluator.get_f1_score()
    conf_matrix = evaluator.get_confusion_matrix()

    print(f"precision score: {prec_score}")
    print(f"f1 score: {f1}")
    print(f"confusion matrix: {conf_matrix}")

    # print(f"images shape: {images.shape} \n labels shape: {labels.shape}")
