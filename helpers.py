import functools
import time
from confluent_kafka import Producer
from datetime import datetime

from constants import PROMPT_LOG_FILENAME


def buffer_too_full(producer: Producer) -> bool:
    """
    Returns True if the producer's buffer is full according to our policy
    """
    return producer.flush(0) > 1


def pprinter(program_name: str, tag: str = None):
    """
    Binds @program_name to print statement for cleaner print statements
    """
    first_col = 17
    if tag:
        program_name += f'/{tag}'
        first_col += 17

    def printer(text):
        print(
            f'[{program_name:{first_col}} |{datetime.utcnow().strftime("%I:%M:%S.%f")}]:{text}'
        )

    return printer


# TODO: is *args necessary?
def rgetattr(obj, attr, *args):
    """
    Takes a string attr1.attr2.(...).attrN and returns the bottom attribute
    source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def append_to_tsv(text_source: str, prompt: str, label: str,
                  pred: str) -> None:

    PROMPT_OUT_LEN = 50
    # Make prompt text exactly 50 chars
    if len(prompt) > PROMPT_OUT_LEN:
        prompt = "..." + prompt[-(PROMPT_OUT_LEN - 3):]
    elif len(prompt) < PROMPT_OUT_LEN:
        prompt = prompt + " " * (PROMPT_OUT_LEN - len(prompt))

    with open(PROMPT_LOG_FILENAME, 'a+') as f:
        f.write(f'{text_source}\t{prompt}\t{label}\t{pred}\n')
