import copy
import pickle
from time import time

from discopy import Word
from discopy.grammar import brute_force
import noisyopt
import numpy as np


def generate_sentences(vocab, n_sentences):
    """
    Generate grammatically correct sentences from vocab.
    The number of sentences is specified by n_sentences.
    """
    gen = brute_force(*vocab)
    sentences, parsing = list(), dict()

    for _ in range(n_sentences):
        diagram = next(gen)
        sentence = ' '.join(str(w)
            for w in diagram.boxes if isinstance(w, Word)) + '.'
        sentences.append(sentence)
        parsing.update({sentence: diagram})

    return sentences, parsing


def regularize_corpus(corpus, threshold=0.5, delta=0.01):
    """
    Regularize corpus truth value assignments
    """
    corpus = copy.copy(corpus)

    for k, v in corpus.items():
        if v > threshold + delta:
            corpus[k] = 1
        elif v < threshold - delta:
            corpus[k] = 0
        else:
            corpus[k] = threshold

    return corpus


def regularize_vector(vector, threshold=0.5, delta=0.01):
    """
    Regularize probability vector
    """
    arr = np.array(vector)

    arr[arr > threshold + delta] = 1
    arr[arr < threshold - delta] = 0
    arr[(arr >= threshold - delta) & (arr <= threshold + delta)] = threshold

    return arr


def subset_corpus(corpus, sentences):
    """
    Acquire a subset of the corpus specified by a set of sentences
    """
    return {st: corpus[st] for st in sentences}


def validate(evaluate, result, corpus, sentences, threshold=0.5, delta=0.01):
    """
    Validate training result
    """
    pred = evaluate(result.x, sentences)
    lbl = list(subset_corpus(corpus, sentences).values())
    truth = regularize_vector(pred, threshold, delta) == regularize_vector(lbl, threshold, delta)
    return truth.mean(), truth, pred, lbl


def print_predictions(corpus, threshold=0.5, delta=0.01):
    """
    Print predictions given truth value assignments to sentences in a corpus
    """
    print('True sentences:\n{}\n'.format('\n'.join('{} ({:.3f})'.format(sentence, output)
        for sentence, output in corpus.items() if output > threshold + delta)))
    print('False sentences:\n{}\n'.format('\n'.join('{} ({:.3f})'.format(sentence, output)
        for sentence, output in corpus.items() if output < threshold - delta)))
    print('Maybe sentences:\n{}\n'.format('\n'.join('{} ({:.3f})'.format(sentence, output)
        for sentence, output in corpus.items() if output <= threshold + delta and output >= threshold - delta)))


def optimize(loss, params, sentences, corpus, **kwargs):
    """
    Invoke SPSA to optimize the params
    """
    global _iter
    _iter = 0
    start = time()

    def callback(params):
        global _iter
        _iter += 1
        print("Iter {} ({:.0f}s): {}".format(_iter, time() - start, params))

    return noisyopt.minimizeSPSA(
        lambda p: loss(p, sentences, corpus),
        params, paired=False, callback=callback, **kwargs
    )


def load(path):
    """
    Load pickled Python object from a file path.
    """
    return pickle.load(open(path, 'rb'))


def save(obj, path):
    """
    Pickle and save Python object to a file path.
    """
    pickle.dump(obj, open(path, 'wb'))
