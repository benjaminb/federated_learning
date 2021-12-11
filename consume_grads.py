from multiprocessing import Pipe
import pickle
import torch
from collections import defaultdict
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.serialization import StringDeserializer
from typing import DefaultDict, List

from constants import BATCH_SIZE, GRAD_TOPIC_NAME, HIDDEN_SIZE, SERVER_MODEL_FILENAME, STEPS_FOR_NEW_MODEL
from helpers import pprinter
from lstm_model import LSTM

PROGRAM_NAME = 'consume_grads.py'

printer = pprinter(PROGRAM_NAME)


def run_grad_consumer(conn: Pipe) -> None:
    settings = {
        'bootstrap.servers': 'localhost:9092',
        # 'bootstrap.servers': '192.168.86.22:9092',
        'group.id': 'grad-consumer-group',  # Gotta specify the group id
        'client.id': 'the-first-client',  # optional
        'enable.auto.commit': True,  # let the consumer auto-report its offset
        'session.timeout.ms':
        6000,  # failure detection protocol shuts this consumer down after this timeout elapses
        'default.topic.config': {
            'auto.offset.reset':
            'smallest'  # start reading from earliest topic events
        },
        # 'fetch.message.max.bytes': 15000000,
        # 'reconnect.backoff.ms': 15000
    }
    model_config = {'hidden_size': HIDDEN_SIZE, 'tokenizer': None}

    # Instantiate a model & send to model producer to distribute
    model = LSTM(**model_config)
    torch.save(model, open(SERVER_MODEL_FILENAME, 'wb'))
    conn.send(1)
    printer("Initial model created, sending to model producer...")

    # Get the model layer names
    layer_names = [p[0] for p in model.named_parameters()]

    # Use the settings above to instantiate the consumer
    c = Consumer(settings)
    deserialize_str = StringDeserializer()

    # Tell this consumer which topics to subscribe to
    # NOTE: subsequent calls will OVERWRITE previous subscriptions!
    c.subscribe([GRAD_TOPIC_NAME])
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

        printer("Consumer started...")
        while True:
            msg = c.poll(1)

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
                    printer(
                        f"Received {BATCH_SIZE} gradients, updating model...")
                    model.update(gradient_dict)
                    model.step_counter += 1
                    printer(f"model updated")

                    # Reset gradient dictionary
                    gradient_dict = defaultdict(list)

                    # Push model back onto pipe
                    if model.step_counter >= STEPS_FOR_NEW_MODEL:
                        torch.save(model, open(SERVER_MODEL_FILENAME, 'wb'))
                        conn.send(1)

            # Case: KafkaError that we reached EOF for this partition
            elif msg.error().code() == KafkaError._PARTITION_EOF:
                printer(
                    f"End of partition reached (topic/partition): {msg.topic()}/{msg.partition()}"
                )
            # Case: some other error
            else:
                printer(f"Error occured: {msg.error().str()}")
    except KeyboardInterrupt:
        c.close()
        return

    finally:
        c.close()