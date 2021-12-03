import torch
from transformers import BertTokenizerFast


class LSTM_NWP(torch.nn.Module):
    def __init__(self, hidden_size: int, tokenizer):
        super().__init__()
        self.hidden_size = hidden_size
        self.tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased') if tokenizer is None else tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        # Define layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        self.lstm = torch.nn.LSTM(input_size=self.hidden_size,
                                  hidden_size=self.hidden_size,
                                  batch_first=True)
        # input_size is hidden_size*2 since we're concatenating hn and cn
        self.linear = torch.nn.Linear(in_features=self.hidden_size * 2,
                                      out_features=self.vocab_size)
        # Initialize weights per suggestion
        torch.nn.init.constant_(self.embedding.weight, val=0)

    def encode_prompt(self, prompt: str) -> torch.tensor:
        '''Takes a string and preps a sample where the last word is the label'''
        #NOTE: encode() returning a list, not a tensor
        prompt_tokenized = self.tokenizer.encode(prompt)

        # Put in list bc forward() expects a batch
        return torch.tensor([prompt_tokenized])

    def forward(self, prompt: str) -> torch.tensor:
        # Encode & embed
        x = self.embedding(self.encode_prompt(prompt))

        # LSTM
        _, (hn, cn) = self.lstm(x)
        hn, cn = torch.squeeze(hn), torch.squeeze(cn)
        x = torch.cat((hn, cn), dim=0)

        # Linear
        x = self.linear(x)
        x = torch.squeeze(x)  #bc it's a single sample
        return x