import pickle
from typing import List


def pickle_brown_sentences(categories: List[str], outfile: str) -> None:
    '''Takes list of categories from Brown corpus (https://www.nltk.org/book/ch02.html), converts
       to dictionary {category: [sentences]}, and pickles to outfile.'''
    from nltk.corpus import brown

    # Creates dictionary and strips off last char from each sentence, which is usually '.'
    sentences = {
        cat: [sent[:-1] for sent in brown.sents(categories=cat)]
        for cat in categories
    }

    with open(outfile, 'wb') as f:
        pickle.dump(sentences, f)
