import torch 
from torch.utils.data import Subset, random_split 
from torchvision.datasets import ImageFolder 
from torchvision import transforms
import json 
from .configs import paths

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(30),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

def load_dataset(folder_path: str, transform): 
    dataset = ImageFolder(root=folder_path, transform=transform)
    return dataset

def split_dataset(dataset, train_fraction): 
   
    train_size = int(train_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = (
        random_split(dataset, [train_size, val_size])
    )

    return train_set, val_set 

def save_preprocessed_data(train_set, val_set): 
    train_indices, val_indices = train_set.indices, val_set.indices
    train_indices_path = paths.PREPROCESSED_DATA_DIR / "train_indices.json"
    val_indices_path = paths.PREPROCESSED_DATA_DIR / "val_indices.json"

    with open(train_indices_path, "w") as f: 
        json.dump(train_indices, f)
    
    with open(val_indices_path, "w") as f: 
        json.dump(val_indices, f) 

    print("your data has been saved successfully")

def load_preprocessed(dataset, train_index_path: str, val_index_path: str):
    train_indices = json.load(open(train_index_path, "r")) 
    val_indices = json.load(open(val_index_path, "r")) 

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print("your dataset has been loaded successfully")
    return train_set, val_set


if __name__ == "__main__": 
    # data_path = input("enter the path to your data: ")
    dataset = load_dataset(folder_path=paths.DATA_DIR, transform=transform)

    train_set, val_set = split_dataset(dataset=dataset, train_fraction=0.8)

    save_preprocessed_data(train_set=train_set, val_set=val_set)
    train_indices_path = paths.PREPROCESSED_DATA_DIR / "train_indices.json"
    val_indices_path = paths.PREPROCESSED_DATA_DIR / "val_indices.json"

    loaded_train_set, loaded_val_set = load_preprocessed(dataset=dataset, train_index_path=train_indices_path, val_index_path=val_indices_path)

    print("passes all sucessfully")

    