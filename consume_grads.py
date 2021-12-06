import pickle
import torch
from collections import defaultdict
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.serialization import StringDeserializer
from typing import DefaultDict, List
from lstm_model import LSTM
from constants import BATCH_SIZE

settings = {
    'bootstrap.servers': 'localhost:9092',  # Gotta specify the kafka cluster
    'group.id': 'grad-consumer-group',  # Gotta specify the group id
    'client.id': 'the-first-client',  # optional
    'enable.auto.commit': True,  # let the consumer auto-report its offset
    'session.timeout.ms':
    6000,  # failure detection protocol shuts this consumer down after this timeout elapses
    'default.topic.config': {
        'auto.offset.reset':
        'smallest'  # start reading from earliest topic events
    },
    'fetch.message.max.bytes': 15000000
}
model_config = {'hidden_size': 50, 'tokenizer': None}

# Instantiate a model
model = LSTM(**model_config)
layer_names = [p[0] for p in model.named_parameters()]

# Use the settings above to instantiate the consumer
c = Consumer(settings)
deserialize_str = StringDeserializer()

# Tell this consumer which topics to subscribe to
# NOTE: subsequent calls will OVERWRITE previous subscriptions!
c.subscribe(['update-test'])
"""
HELPER FUNCTIONS
"""


def ready_to_update(
        gradient_dict: DefaultDict[str, List[torch.tensor]]) -> bool:
    """
    Checks that gradient_dict has at least BATCH_SIZE updates for all layers.
    """
    return all(
        [len(gradient_dict[layer]) >= BATCH_SIZE for layer in layer_names])


try:
    gradient_dict = defaultdict(list)

    print("Consumer started...")
    while True:
        msg = c.poll(0.1)

        # Case: no new messages so continue
        if msg is None:
            continue

        # Case: messages received and no error
        if not msg.error():
            key = deserialize_str(msg.key(), None)
            value = pickle.loads(msg.value())
            gradient_dict[key] += [value]

            # Update model when all layers have at least BATCH_SIZE gradient updates
            if ready_to_update(gradient_dict):
                print(f"Received {BATCH_SIZE} gradients, updating model...")
                model.update(gradient_dict)
                print(f"model updated")

                # Reset gradient dictionary
                gradient_dict = defaultdict(list)

        # Push model back to devices

        # Case: KafkaError that we reached EOF for this partition
        elif msg.error().code() == KafkaError._PARTITION_EOF:
            print(
                f"End of partition reached (topic/partition): {msg.topic()}/{msg.partition()}"
            )
        # Case: some other error
        else:
            print(f"Error occured: {msg.error().str()}")
except KeyboardInterrupt:
    pass

finally:
    c.close()