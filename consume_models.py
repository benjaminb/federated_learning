from multiprocessing import Pipe
import pickle
import torch
from collections import defaultdict
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.serialization import StringDeserializer
from typing import DefaultDict, List

from constants import BATCH_SIZE
from lstm_model import LSTM
from helpers import pprinter

PROGRAM_NAME = 'consume_models.py'
printer = pprinter(PROGRAM_NAME)


def run_model_consumer(conn: Pipe, consumer_group_name: str) -> None:
    settings = {
        'bootstrap.servers':
        'localhost:9092',  # Gotta specify the kafka cluster
        'group.id': consumer_group_name,  # Gotta specify the group id
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

    # Instantiate a model & send to model producer to distribute
    model = LSTM(**model_config)
    conn.send(model)
    printer(f"Initial model created, sending to model producer...")

    # Get the model layer names
    layer_names = [p[0] for p in model.named_parameters()]

    # Use the settings above to instantiate the consumer
    consumer = Consumer(settings)
    deserialize_str = StringDeserializer()

    consumer.subscribe(['update-model-test'])

    def ready_to_update(weight_dict: DefaultDict[str, torch.tensor]) -> bool:
        """
        Returns True if weight dict has an update for each layer
        """
        return list(weight_dict.keys()) == layer_names

    try:
        weight_dict = defaultdict(None)

        printer("Model consumer started...")
        while True:
            msg = consumer.poll(2)

            # Case: no new messages so continue
            if msg is None:
                continue

            # Case: messages received and no error
            if not msg.error():
                key = deserialize_str(msg.key(), None)
                value = pickle.loads(msg.value())
                weight_dict[key] = value

                if ready_to_update(weight_dict):
                    printer("Ready to update model, rewriting weights...")
                    model.replace_weights(weight_dict)
                    printer("Model weights updated.")

                    # Reset gradient dictionary
                    weight_dict = defaultdict(None)

                    # Push model back onto pipe
                    printer("Sending new model to grad producer...")
                    conn.send(model)

            # Case: KafkaError that we reached EOF for this partition
            elif msg.error().code() == KafkaError._PARTITION_EOF:
                printer(
                    f"End of partition reached (topic/partition): {msg.topic()}/{msg.partition()}"
                )
            # Case: some other error
            else:
                printer(f"Error occured: {msg.error().str()}")
    except KeyboardInterrupt:
        pass

    finally:
        consumer.close()