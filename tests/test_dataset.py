from src.configs.paths import (
    ROOT_DIR, 
    DATA_DIR, 
    PREPROCESSED_DATA_DIR
)

from src.dataset import (
    load_dataset, 
    split_dataset, 
    save_preprocessed_data, 
    load_preprocessed
)

def test_load_dataset(): 
    dataset = load_dataset(DATA_DIR, transform=None)
    assert len(dataset) > 0, "dataset must not be empty"

def test_split_dataset(): 
    dataset = load_dataset(DATA_DIR, transform=None)
    train_set, val_set = split_dataset(dataset=dataset, train_fraction=0.8)
    
    assert len(train_set) + len(val_set) == len(dataset), "length of train and val set must add up"

def test_load_preprocessed(): 
    dataset = load_dataset(DATA_DIR, transform=None)
    train_set, val_set = load_preprocessed(
        train_index_path=str(PREPROCESSED_DATA_DIR) + "/train_indices.json", 
        val_index_path=str(PREPROCESSED_DATA_DIR) + "/val_indices.json", 
        dataset=dataset
    )

    assert len(train_set) + len(val_set) == len(dataset), "lenght of train and val set must add up"