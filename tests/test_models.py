import pytest 
import torch 
from torch import Tensor

from src.model import SimpleTomatoCNN

def test__simple_cnn_type(): 
    model = SimpleTomatoCNN(n_classes=10)
    assert isinstance(model, SimpleTomatoCNN), "model must be an instance of SimpleTomatoCNN"

def test_simple_cnn_forward(): 
    model = SimpleTomatoCNN(n_classes=10)
    sample_data = torch.randn(4, 3, 224, 224)

    output = model(sample_data)
    assert isinstance(output, Tensor), "output of the model must be a Tensor object"

