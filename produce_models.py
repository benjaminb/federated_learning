from multiprocessing import Pipe
import pickle
import time
import torch
from confluent_kafka import Producer
from confluent_kafka.serialization import StringSerializer

from constants import WAIT_TIME_ON_BUFFER_ERROR, MODEL_TOPIC_NAME, SERVER_MODEL_FILENAME
from helpers import pprinter, buffer_too_full

PROGRAM_NAME = "produce_models.py"
printer = pprinter(PROGRAM_NAME)


def run_model_producer(conn: Pipe) -> None:
    print("Starting model producer")

    def ack(err, msg):
        """@err: error thrown by producer
            @msg: the kafka message object"""
        if err:
            printer(f"Failed to deliver message: {msg.key()}\n{err.str()}")
        else:
            # printer(
            #     f"Model layer weight update message produced: {msg.key()}: {len(msg.value()):,} bytes"
            # )
            pass

    printer("Model producer starting...")

    producer_config = {
        'bootstrap.servers': 'kafka:9092',
        'message.max.bytes': 150000000,
        # 'reconnect.backoff.ms': 15000,
        # 'reconnect.backoff.max.ms': 15001,
    }

    producer = Producer(producer_config)
    serialize_str = StringSerializer()

    # Wait for consume_grads to send over initial model
    printer("Model producer is waiting for initial model...")
    _ = conn.poll(timeout=None)
    printer("Model producer received initial model.")

    while True:
        try:
            """
            WAIT FOR MESSAGE
            """
            if not conn.poll(1):
                continue

            # Get the saved model from disk
            _ = conn.recv()  # Clear the pipe queue
            model = pickle.load(open(SERVER_MODEL_FILENAME, 'rb'))
            printer("Model producer received an updated model...")

            # Confirm model has been updated
            # if not model.updated:
            #     continue
            """
            SEND UPDATED MODEL
            """
            printer("Sending model updates to broker...")
            for key, grad in model.named_parameters():
                ser_key = serialize_str(key, ctx=None)
                value = pickle.dumps(grad.data)

                producer.produce(topic=MODEL_TOPIC_NAME,
                                 key=ser_key,
                                 value=value,
                                 callback=ack)
                producer.poll(0)  # Trigger queue cleaning
                model.updated = False
        except BufferError:
            printer("BufferError encountered. Waiting for buffer to clear...")
            while buffer_too_full(producer=producer):
                printer(f"Messages in queue: {producer.flush(0)}")
                time.sleep(WAIT_TIME_ON_BUFFER_ERROR)
        except KeyboardInterrupt:
            producer.flush(30)
            return

    producer.flush(30)