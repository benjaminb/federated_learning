from multiprocessing import Pipe
import pickle
import torch
from confluent_kafka import Producer
from confluent_kafka.serialization import StringSerializer
# from lstm_model import LSTM


def run_model_producer(conn: Pipe) -> None:
    def ack(err, msg):
        """@err: error thrown by producer
            @msg: the kafka message object"""
        if err:
            print(f"Failed to deliver message: {msg.key()}\n{err.str()}")
        else:
            print(
                f"Model layer weight update message produced: {msg.key()}: {len(msg.value()):,} bytes"
            )

    print("Model producer starting...")

    producer_config = {
        'bootstrap.servers': 'localhost:9092',
        'message.max.bytes': 15000000,
    }

    producer = Producer(producer_config)
    serialize_str = StringSerializer()

    # Wait for consume_grads to send over initial model
    print("Model producer is waiting for initial model...")
    _ = conn.poll(timeout=None)
    print("Model producer received initial model.")

    try:
        while True:
            """
            WAIT FOR MESSAGE
            """
            if not conn.poll(1):
                continue

            # Get the model from the pipe
            model = conn.recv()
            print("Model producer received an update...")
            print(f"model.updated={model.updated}")
            # Confirm model has been updated
            if not model.updated:
                continue
            """
            SEND UPDATED MODEL
            """
            for key, grad in model.named_parameters():
                ser_key = serialize_str(key, ctx=None)
                value = pickle.dumps(grad.data)

                producer.produce(topic='update-model-test',
                                 key=ser_key,
                                 value=value,
                                 callback=ack)

                model.updated = False

    except KeyboardInterrupt:
        pass

    p.flush(30)