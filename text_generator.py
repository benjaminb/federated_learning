# Given a list of list of strings (a list of word-tokenized sentences),
# randomly sample a sentence and a cutoff, returning the partial sentence
# and true next word (input, label)

import random
from typing import Tuple, List


class TextGenerator():
    def __init__(self, sentences):
        self.sentences = sentences

    def sample(self) -> Tuple[List[str], str]:
        sentence = []
        # Ensure sentence has at least 2 tokens
        while len(sentence) < 2:
            sentence = random.choice(self.sentences)
        cutoff = random.randint(0, len(sentence) - 1)
        return sentence[:cutoff], sentence[cutoff]


# For testing
if __name__ == '__main__':
    import pickle
    sentences = pickle.load(open('brown.pkl', 'rb')).get('editorial')
    gen = TextGenerator(sentences)

    for i in range(10):
        print(gen.sample())