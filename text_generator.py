# Given a list of list of strings (a list of word-tokenized sentences),
# randomly sample a sentence and a cutoff, returning the partial sentence
# and true next word (input, label)

import random
from typing import Tuple, List


class TextGenerator():
    def __init__(self, sentences):
        self.sentences = sentences

    def generate_samples(self, n: int) -> Tuple[List[str], str]:
        samples = []
        for _ in range(n):
            sentence = []

            # Ensure sentence has at least 2 tokens
            while len(sentence) < 2:
                sentence = random.choice(self.sentences)

            # Choose cutoff & mask it
            cutoff = random.randint(0, len(sentence) - 1)
            masked_sentence = sentence[:cutoff] + ["[MASK]"]
            samples.append((masked_sentence, sentence[cutoff]))
        return samples


# For testing
if __name__ == '__main__':
    import pickle
    sentences = pickle.load(open('brown.pkl', 'rb')).get('editorial')
    gen = TextGenerator(sentences)

    samples = gen.generate_samples(5)
    print(samples)