import torch
from src.model import SimpleTomatoCNN 
from src.dataset import (
    transform, 
    split_dataset, 
    save_preprocessed_data, 
    load_dataset, 
    load_preprocessed
)

from src.configs import paths 
 
from src.train import Trainer
from torch.utils.data import DataLoader



dataset = load_dataset(folder_path=paths.DATA_DIR, transform=transform)
train_set, val_set = split_dataset(dataset=dataset, train_fraction=0.8)

save_preprocessed_data(train_set=train_set, val_set=val_set)
train_indices_path = paths.PREPROCESSED_DATA_DIR / "train_indices.json"
val_indices_path = paths.PREPROCESSED_DATA_DIR / "val_indices.json"

loaded_train_set, loaded_val_set = load_preprocessed(dataset=dataset, train_index_path=train_indices_path, val_index_path=val_indices_path)

# print("data loaded and preprocessed sucessfully")

# print(type(loaded_train_set))
# print(type(loaded_val_set))


# apply transformations and convert data to dataloaders
train_loader = DataLoader(
    loaded_train_set, 
    batch_size=32, 
    shuffle=True
)

val_loader = DataLoader(
    loaded_val_set,
    batch_size=32
)

model = SimpleTomatoCNN(n_classes=10)

# device = "cuda" if torch.cuda.is_available() else "cpu"

trainer = Trainer(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    # device=device
)

# training 
trainer.train_model(epochs=2, lr=0.1)


# save model 
trainer.save_model()
