# Load data
# load in data from pickle
# create Dataset
# create DataLoader
# create model
# define training step, produce loss

import string, torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM
from typing import List, Tuple

from text_generator import TextGenerator

top_k = 1
# Get raw data and samples from text generator

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')


# Code extracted and adapted from Jovian module
def encode(tokenizer, text_sentence, add_special_tokens=True):
    # Sentence is already masked last word so this line not needed
    # text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)

    # NOTE: this should always be the case
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

        input_ids = torch.tensor([
            tokenizer.encode(text_sentence,
                             add_special_tokens=add_special_tokens)
        ])

        mask_idx = torch.where(
            input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def get_all_predictions(text_sentence, top_clean=5):
    input_ids, mask_idx = encode(tokenizer, text_sentence)
    with torch.no_grad():
        predict = model(input_ids)[0]
    preds = decode(tokenizer,
                   predict[0, mask_idx, :].topk(top_k).indices.tolist(),
                   top_clean)
    return preds


def get_prediction_eos(input_text):
    try:
        input_text += ' ' + tokenizer.mask_token
        # TODO: parametrize top_k better?
        res = get_all_predictions(input_text, top_clean=int(top_k))
        return res
    except Exception as error:
        print(f"An exception occured\n{error}")


# For testing
if __name__ == '__main__':
    # Full sentence, no mask yet
    sentence = "This is a test of the emergency broadcast"

    # Load model
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    result = get_prediction_eos(sentence)
    print(result)