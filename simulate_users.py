import multiprocessing
from multiprocessing import Process, Pipe
from produce_grads import run_grad_producer
from consume_models import run_model_consumer

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    consumer_pipe, producer_pipe = Pipe()

    producer_proc = Process(target=run_grad_producer,
                            args=(
                                producer_pipe,
                                'treasureisland.txt',
                            ))

    consumer_proc = Process(target=run_model_consumer,
                            args=(consumer_pipe, 'treasureisland-consumer-gp'))

    # Set up secret garden
    cons2_pipe, prod2_pipe = Pipe()
    producer2_proc = Process(target=run_grad_producer,
                             args=(prod2_pipe, 'secretgarden.txt'))
    consumer2_proc = Process(target=run_model_consumer,
                             args=(cons2_pipe, 'secretgarden-consumer-gp'))

    consumer_proc.start()
    producer_proc.start()
    consumer2_proc.start()
    producer2_proc.start()

    producer_proc.join()
    consumer_proc.join()
    producer2_proc.join()
    consumer2_proc.join()