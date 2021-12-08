from multiprocessing import Pipe
import os
import pickle
import time
import torch
from confluent_kafka import Producer
from confluent_kafka.serialization import StringSerializer

from constants import HIDDEN_SIZE, CREATE_SAMPLE_INTERVAL, GRAD_TOPIC_NAME, PROMPT_LOG_INTERVAL, WAIT_TIME_ON_BUFFER_ERROR, PATH_TO_DATA
from lstm_model import LSTM
from text_generator import TextGenerator
from helpers import pprinter, append_to_tsv, buffer_too_full

PROGRAM_NAME = 'produce_grads.py'


def run_grad_producer(conn: Pipe, text_source: str):
    # Acknowledgement callback
    def ack(err, msg):
        """@err: error thrown by producer
            @msg: the kafka message object"""
        if err:
            printer(f"Failed to deliver message: {msg.key()}\n{err.str()}")
        else:
            # printer(
            #     f"Gradient message produced: {msg.key()}: {len(msg.value()):,} bytes"
            # )
            pass

    # Resolve path to text source
    path_to_text = os.path.join(PATH_TO_DATA, text_source)
    assert os.path.exists(
        path_to_text), f"PROBLEM: {path_to_text} does not exist"

    printer = pprinter(PROGRAM_NAME, tag=text_source)

    def label_to_tensor(text: str) -> torch.LongTensor:
        text_ids = tokenizer.convert_tokens_to_ids(text)
        return torch.LongTensor([text_ids])

    # Configs
    model_config = {'hidden_size': HIDDEN_SIZE, 'tokenizer': None}
    text_gen_config = {'path_to_text': path_to_text}
    producer_config = {
        'bootstrap.servers': 'localhost:9092',
        'message.max.bytes': 15000000
    }

    # Set up model
    model = LSTM(**model_config)
    loss_fn = torch.nn.CrossEntropyLoss()
    tokenizer = model.tokenizer

    # Set up sample text generator
    text_gen = TextGenerator(**text_gen_config)

    # Instantiate a producer & serializers
    p = Producer(producer_config)
    serialize_str = StringSerializer()

    # Get references to the gradient tensors
    named_params = list(model.named_parameters())

    prompt_log_counter = 0
    while True:
        try:
            # Check if there's a new model
            if conn.poll():
                model = conn.recv()
                printer("New model received")
            """
            SIMULATE USAGE
            """
            time.sleep(CREATE_SAMPLE_INTERVAL)
            # Get a prompt
            prompt, label = text_gen.generate_sample()
            target = label_to_tensor(label)

            # Compute loss & get gradient
            logits = model(prompt)
            loss = loss_fn(logits, target)
            loss.backward()

            prompt_log_counter += 1

            # Log the results according to the interval constant
            if prompt_log_counter == PROMPT_LOG_INTERVAL:
                pred = model.decode_logits(logits)
                append_to_tsv(text_source=text_source,
                              prompt=prompt,
                              label=label,
                              pred=pred)
                printer("Wrote to prompt log file...")
                prompt_log_counter = 0
            """
            SEND UPDATES TO KAFKA
            """
            for key, grad in named_params:
                # TODO: keys don't change so store them in a table
                ser_key = serialize_str(key, ctx=None)
                value = pickle.dumps(grad.data)
                # Do the produce call
                p.produce(topic=GRAD_TOPIC_NAME,
                          key=ser_key,
                          value=value,
                          callback=ack)

                p.poll(0)
        except BufferError:
            printer("BufferError encountered. Waiting for buffer to clear...")
            while buffer_too_full(producer=p):
                printer(f"Messages in queue: {p.flush(0)}")
                time.sleep(WAIT_TIME_ON_BUFFER_ERROR)
        except KeyboardInterrupt:
            p.flush(30)
            return

    p.flush(30)
