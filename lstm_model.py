import time
import torch
from transformers import BertTokenizerFast
from typing import DefaultDict, List
from helpers import rgetattr


class LSTM(torch.nn.Module):
    def __init__(self, hidden_size: int, tokenizer=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased') if tokenizer is None else tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.updated = False
        self.step_counter = 0

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
        return x.view(1, -1)  #[1, vocab_size]

    def update(self,
               grad_dict: DefaultDict[str, List[torch.tensor]],
               lr=0.01) -> None:
        '''Updates the weights of the model'''
        with torch.no_grad():
            for layer, grad_list in grad_dict.items():
                # Average the gradient
                batch_gradient = torch.mean(torch.stack(grad_list), dim=0)

                # Resolve the layer and perform gradient descent
                parameter = rgetattr(self, layer)
                parameter.data -= lr * batch_gradient

        self.updated = True
        self.last_updated = time.time()

    def replace_weights(self, weight_dict: DefaultDict[str,
                                                       torch.tensor]) -> None:
        with torch.no_grad():
            for layer, new_weights in weight_dict.items():
                parameter = rgetattr(self, layer)
                parameter.data = new_weights

    def decode_logits(self, inp) -> str:
        """
        Decodes input tensor from a logits prediction
        """
        return self.tokenizer.decode(torch.argmax(inp))

    def label_to_tensor(self, text: str) -> torch.LongTensor:
        """
        Converts text string to torch.LongTensor using tokenizer
        """
        text_ids = self.tokenizer.convert_tokens_to_ids(text)
        return torch.LongTensor([text_ids])