import multiprocessing
import os
from multiprocessing import Process, Pipe
from produce_grads import run_grad_producer
from consume_models import run_model_consumer
from constants import PATH_TO_DATA, TEXT_SOURCES


def main():
    multiprocessing.set_start_method('spawn')

    text_sources = TEXT_SOURCES if TEXT_SOURCES else os.listdir(PATH_TO_DATA)
    pipes = [Pipe() for _ in range(len(text_sources))]
    consumers, producers = [], []

    for i, ((c_pipe, p_pipe),
            text_source) in enumerate(zip(pipes, text_sources)):
        producer = Process(target=run_grad_producer,
                           args=(p_pipe, text_source, i))
        consumer = Process(target=run_model_consumer,
                           args=(c_pipe, f'{text_source[:-4]}-consumer-gp'))

        producers.append(producer)
        consumers.append(consumer)

        consumer.start()
        producer.start()

    # Start log file for prompts
    with open('prompts.txt', 'w') as f:
        f.write('TEXT\tPROMPT\tLABEL\tPREDICTION\n')

    for producer, consumer in zip(producers, consumers):
        producer.join()
        consumer.join()


if __name__ == '__main__':
    main()
