"""Example of a multiclass active learning text classification with pytorch.
"""
import logging

import torch
import numpy as np

from pathlib import Path

from small_text.active_learner import PoolBasedActiveLearner
from small_text.exceptions import ActiveLearnerException
from small_text.initialization import random_initialization_stratified
from small_text.integrations.pytorch.classifiers.factories import KimCNNFactory
from small_text.integrations.pytorch.query_strategies import ExpectedGradientLength
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException

from examples.data.example_data_multiclass import get_train_test
from examples.shared import evaluate

try:
    import gensim.downloader as api
except ImportError:
    raise ActiveLearnerException('This example requires the gensim library. '
                                 'Please install gensim 3.8.x to run this example.')


def main():

    device = 'cuda'
    path = Path('.data/')

    if not path.exists():
        path.mkdir()

    # Prepare some data
    train, test = get_train_test()
    num_classes = len(np.unique(train.y))

    # Active learning parameters
    classifier_kwargs = dict({'embedding_matrix': _load_gensim_embedding(train.vocab),
                              'device': device})

    clf_factory = KimCNNFactory('kimcnn', num_classes, classifier_kwargs)
    query_strategy = ExpectedGradientLength(num_classes, device=device)

    # Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)

    labeled_indices = random_initialization_stratified(train.y, 20)
    y_initial = train[labeled_indices].y

    active_learner.initialize_data(labeled_indices, y_initial)

    try:
        # Perform 20 iterations of active learning...
        for i in range(20):
            # ...where each iteration consists of labelling 20 samples
            q_indices = active_learner.query(num_samples=20, x=train)

            # Simulate user interaction here. Replace this for real-world usage.
            y = train.y[q_indices]

            # Return the labels for the current query to the active learner.
            active_learner.update(y)

            labeled_indices = np.concatenate([q_indices, labeled_indices])

            print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
            evaluate(active_learner, train[labeled_indices], test)

    except PoolExhaustedException:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException:
        print('Error! No more samples left. (Unlabeled pool is empty)')


def _load_gensim_embedding(vocab, min_freq=1):
    # assert vocab.itos[0] == '<pad>'
    pretrained_vectors = api.load('word2vec-google-news-300')

    vectors = [
        np.zeros(pretrained_vectors.vectors.shape[1]),  # <ukn>
        np.zeros(pretrained_vectors.vectors.shape[1])   # <pad>
    ]
    num_special_vectors = len(vectors)
    vectors += [
        pretrained_vectors.vectors[pretrained_vectors.vocab[vocab.itos[i]].index]
        if vocab.itos[i] in pretrained_vectors.vocab
        else np.zeros(pretrained_vectors.vectors.shape[1])
        for i in range(num_special_vectors, len(vocab))
    ]
    for i in range(num_special_vectors, len(vocab)):
        if vocab.itos[i] not in pretrained_vectors.vocab and vocab.freqs[vocab.itos[i]] >= min_freq:
            vectors[i] = np.random.uniform(-0.25, 0.25, pretrained_vectors.vectors.shape[1])

    return torch.as_tensor(np.stack(vectors))


if __name__ == '__main__':
    logging.getLogger('small_text').setLevel(logging.INFO)
    main()
