import numpy as np
import pickle
import torch
from confluent_kafka import Producer
from confluent_kafka.serialization import StringSerializer
from transformers import BertTokenizerFast
from lstm_model import LSTM
from text_generator import TextGenerator


# Acknowledgement callback
def ack(err, msg):
    """@err: error thrown by producer
        @msg: the kafka message object"""
    if err:
        print(f"Failed to deliver message: {msg.key()}\n{err.str()}")
    else:
        print(f"Message produced")


#TODO: Get the tokenizer from the model instance instead?
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def label_to_tensor(text: str) -> torch.LongTensor:
    text_ids = tokenizer.convert_tokens_to_ids(text)
    return torch.LongTensor([text_ids])


# Configs
model_config = {'hidden_size': 50, 'tokenizer': None}
text_gen_config = {'path_to_text': 'text_sources/treasureisland.txt'}
producer_config = {
    'bootstrap.servers': 'localhost:9092',
    'message.max.bytes': 15000000
}

# Set up model
model = LSTM(**model_config)
loss_fn = torch.nn.CrossEntropyLoss()

# Set up sample text generator
text_gen = TextGenerator(**text_gen_config)

# Instantiate a producer & serializers
p = Producer(producer_config)
serialize_str = StringSerializer()

# Get the string names of the model's layers (these are constant)
keys = model.state_dict().keys()

# Get references to the gradient tensors
named_params = list(model.named_parameters())

try:
    for val in range(4):
        # Get a prompt
        prompt, label = text_gen.generate_sample()
        target = label_to_tensor(label)

        # Compute loss & get gradient
        logits = model(prompt)
        loss = loss_fn(logits, target)
        loss.backward()
        for key, grad in named_params:
            pickled = pickle.dumps(grad.data)
            print(f"Pickled {len(pickled)} bytes for {key}")
            # Do the produce call
            p.produce(topic='update-test',
                      key=serialize_str(key, ctx=None),
                      value=pickled,
                      callback=ack)

            p.poll(0.5)
except KeyboardInterrupt:
    pass

p.flush(30)
