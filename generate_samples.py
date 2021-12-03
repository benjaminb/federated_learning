from typing import Tuple


def parse_sample(self, text: str) -> Tuple[str, str]:
    '''Takes a string and parses it into a tuple (prompt, label) where last word is label, 
        remainder is the input'''
    prompt, label = text.rsplit(sep=' ', maxsplit=1)
    return prompt, label