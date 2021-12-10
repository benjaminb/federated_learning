"""
TEST MODEL
"""
import torch
from lstm_model import LSTM
from transformers import BertTokenizerFast

model = LSTM(10)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


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
