import multiprocessing
from multiprocessing import Process, Pipe
from consume_grads import run_grad_consumer
from produce_models import run_model_producer

if __name__ == '__main__':
    # Should support both *nix and windows
    # https://docs.python.org/3/library/multiprocessing.html
    multiprocessing.set_start_method('spawn')
    consumer_pipe, producer_pipe = Pipe()

    consumer_proc = Process(target=run_grad_consumer, args=(consumer_pipe, ))
    consumer_proc.start()

    producer_proc = Process(target=run_model_producer, args=(producer_pipe, ))
    producer_proc.start()

    consumer_proc.join()
    producer_proc.join()