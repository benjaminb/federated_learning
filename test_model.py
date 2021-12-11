"""
TEST MODEL
"""
import torch
from collections import defaultdict
from helpers import rgetattr
from lstm_model import LSTM
from transformers import BertTokenizerFast

model = LSTM(10)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
layer_names = [p[0] for p in model.named_parameters()]


def test_model_encode_prompt():
    prompt = "hello world"
    result = model.encode_prompt(prompt)
    expected = torch.tensor([tokenizer.encode(prompt)])
    assert torch.equal(expected, result)


def test_model_forward():
    """Given a string prompt, should produce a Long int 1D tensor, same length as the tokenizer's vocabulary"""
    vocab_size = tokenizer.vocab_size
    prompt = "hello world"
    result = model.forward(prompt)
    assert result.shape.numel() == vocab_size


def test_model_update():
    '''Given a defaultdict of layer_name: tensor, should update the model's weights'''
    # Create ones tensors for all layers
    grad_dict = defaultdict(list)
    old_weights = [torch.clone(param.data) for param in model.parameters()]
    for layer_name in layer_names:
        param = rgetattr(model, layer_name)
        shape = param.shape
        old_weights.append(torch.clone(param.data))
        grad_dict[layer_name] = [torch.ones(shape)] * 2

    model.update(grad_dict, lr=1)
    new_weights = [param.data for param in model.parameters()]

    # Check that weights have changed
    for old, new in zip(old_weights, new_weights):
        assert not torch.equal(old, new)


def test_decode_logits():
    '''Expects [PAD] token to be decoded'''
    logits = torch.rand(model.tokenizer.vocab_size)

    # Force logits[0] to be the argmax
    logits[0] += 1
    token = model.decode_logits(logits)
    assert token == '[PAD]'