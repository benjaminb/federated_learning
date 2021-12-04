from lstm_model import LSTM
import torch
import pickle
from transformers import BertTokenizerFast

model = LSTM(hidden_size=100, tokenizer=None)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Get a sample
prompt = "hello how are"

# Get tokenizer id for label & tensorize
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
label = tokenizer.convert_tokens_to_ids("you")
label = torch.LongTensor([label])

# Get the loss
logits = model(prompt)
logits = logits.reshape(1, -1)
loss = loss_fn(logits, label)
loss.backward()

print("accessing by state dict")
state_dict = model.state_dict()
print(f"Gradient on linear: {state_dict['linear.weight'].grad}")
print(model.linear.weight.grad.shape)

print(f"Shape of linear: {state_dict['linear.weight'].shape}")

print('Output of model.parameters():')
params = model.parameters()
print(len(params))
print(type(params))
# with open('state_dict.txt', 'w') as f:
#     f.write(model.state_dict().__repr__())

# with open('named_params.txt', 'w') as f:
#     f.write(model.named_parameters().__repr__())