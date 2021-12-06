import multiprocessing
from produce_grads import simulate_producer

if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')
    produce_proc = ctx.Process(target=simulate_producer,
                               args=('text_sources/treasureisland.txt', ))

    produce_proc.start()
    produce_proc.join()