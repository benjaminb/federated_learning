# Given a list of list of strings (a list of word-tokenized sentences),
# randomly sample a sentence and a cutoff, returning the partial sentence
# and true next word (input, label)

import nltk, random
import string
from typing import Tuple, List


def text_to_sentences(text: str) -> List[str]:
    """
    Given a string of text, tokenize into sentences of at least 2 words
    and return a list of sentences.
    """
    def good_sentence(s):
        """
        Predicate for filtering sentences since nltk.word_tokenize treats
        punctuation as a word.
        """
        stripped = ''.join([c for c in s if c not in string.punctuation])
        words = nltk.word_tokenize(stripped)
        return len(words) > 3

    sents = [s for s in nltk.sent_tokenize(text) if good_sentence(s)]
    sents = [s.replace('\n', ' ') for s in sents]
    return sents


class PromptGenerator():
    def __init__(self, text):
        self.sentences = text_to_sentences(text)

    def generate_sample(self) -> Tuple[str, str]:
        """
        Since the last token in a sentence will almost always be punctuation,
        we use the -2 index as the target and text[:-2] as the prompt.
        """
        sentence = None
        while not sentence:
            sentence = random.choice(self.sentences)
            # Ensure sentence has at least 3 tokens
            words = nltk.word_tokenize(sentence)
            assert len(
                words
            ) > 2, "Sentence too short in PromptGenerator.generate_sample"

        print(f"Chosen sentence: {sentence}")
        # Choose cutoff & mask it
        cutoff = random.randint(0, len(words) - 2)  # -2 to avoid punctuation
        label = words[cutoff]
        prompt = sentence[:sentence.rfind(label)]
        return prompt, label

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