# Load data
# load in data from pickle
# create Dataset
# create DataLoader
# create model
# define training step, produce loss

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast
from typing import List, Tuple

from text_generator import TextGenerator

# Get raw data and samples from text generator

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


class SentenceDataset(Dataset):
    def __init__(self, sequences: List[List[str]], labels: List[str]):
        self.sentences = [tokenizer(seq).get('input_ids') for seq in sequences]
        self.labels = tokenizer(labels).get('input_ids')

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        return (torch.tensor(self.sentences[idx]),
                torch.tensor(self.labels[idx]))


# For testing
if __name__ == '__main__':
    sequences = [['this', 'is', 'a'], ['the', 'emergency', 'broadcast']]
    labels = ['test', 'system']
    dataset = SentenceDataset(sequences, labels)
    print(f"Len of dataset: {len(dataset)}")
    print(f"First item: {dataset[0]}")
