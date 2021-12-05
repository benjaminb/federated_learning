import pickle
import torch
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.serialization import StringDeserializer
from lstm_model import LSTM

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
    }
}

# Use the settings above to instantiate the consumer
c = Consumer(settings)
deserialize_str = StringDeserializer()

# Tell this consumer which topics to subscribe to
# NOTE: subsequent calls will OVERWRITE previous subscriptions!
c.subscribe(['grads-test'])

try:
    while True:
        msg = c.poll(0.1)

        # Case: no new messages so continue
        if msg is None:
            continue

        # Case: messages received and no error
        if not msg.error():
            key = deserialize_str(msg.key(), None)
            value = pickle.loads(msg.value())

        # Store the gradients somewhere by key
        # in a defaultdict, value defaults to empty list
        # for each key, value, update defaultdict key: oldvalue + [new value]
        # so we get a list of grads for each key: [grad1, ..., grad32]
        # When BATCH_SIZE=32 has been reached, average them & apply LR
        # Then reset the gradients to update model
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