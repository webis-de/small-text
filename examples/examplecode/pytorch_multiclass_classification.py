"""Example of a multiclass active learning text classification with pytorch.

Note:
This examples requires gensim 3.8.x which is used for obtaining word2vec embeddings.
"""
import torch
import numpy as np

from collections import Counter

from small_text import (
    ActiveLearnerException,
    EmptyPoolException,
    ExpectedGradientLength,
    KimCNNFactory,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    random_initialization_stratified
)

from examplecode.data.example_data_multiclass import (
    get_train_test,
    preprocess_data
)
from examplecode.shared import evaluate

try:
    import gensim.downloader as api
except ImportError:
    raise ActiveLearnerException('This example requires the gensim library. '
                                 'Please install gensim 3.8.x to run this example.')


def main(num_iterations=10, device='cuda'):
    pretrained_vectors = api.load('word2vec-google-news-300')

    # Prepare some data
    train, test = get_train_test()

    # TODO: use another dataset
    train, test, tokenizer = preprocess_data(train, test, pretrained_vectors)

    num_classes = len(np.unique(train.y))

    # Active learning parameters
    # TODO: the selection of embedding vectors can still be improved
    classifier_kwargs = {
        'embedding_matrix': load_gensim_embedding(train.data, tokenizer, pretrained_vectors),
        'max_seq_len': 512,
        'num_epochs': 4,
        'device': device
    }

    # TODO: clean up KimCNNFactory
    clf_factory = KimCNNFactory('kimcnn', num_classes, classifier_kwargs)
    query_strategy = ExpectedGradientLength(num_classes, device=device)

    # Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
    indices_labeled = initialize_active_learner(active_learner, train.y)

    try:
        perform_active_learning(active_learner, train, indices_labeled, test, num_iterations)

    except PoolExhaustedException:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException:
        print('Error! No more samples left. (Unlabeled pool is empty)')


def perform_active_learning(active_learner, train, indices_labeled, test, num_iterations):
    # Perform 20 iterations of active learning...
    for i in range(num_iterations):
        # ...where each iteration consists of labelling 20 samples
        indices_queried = active_learner.query(num_samples=20, representation=train)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print('Iteration #{:d} ({} samples)'.format(i, len(indices_labeled)))
        evaluate(active_learner, train[indices_labeled], test)


def load_gensim_embedding(texts, tokenizer, pretrained_vectors, min_freq=1, num_special_tokens=2):

    vectors = [
        np.zeros(pretrained_vectors.vectors.shape[1])
        for _ in range(num_special_tokens)
    ]
    vocab = tokenizer.get_vocab()
    vectors += [
        pretrained_vectors.vectors[pretrained_vectors.vocab[tokenizer.id_to_token(i)].index]
        if tokenizer.id_to_token(i) in pretrained_vectors.vocab
        else np.zeros(pretrained_vectors.vectors.shape[1])
        for i in range(num_special_tokens, len(vocab))
    ]

    token_id_list = [text[0].cpu().numpy().tolist() for text in texts]
    word_frequencies = Counter([token for tokens in token_id_list for token in tokens])
    for i in range(num_special_tokens, len(vocab)):
        is_in_vocab = tokenizer.id_to_token(i) not in pretrained_vectors.vocab
        if is_in_vocab and word_frequencies[tokenizer.id_to_token(i)] >= min_freq:
            vectors[i] = np.random.uniform(-0.25, 0.25, pretrained_vectors.vectors.shape[1])

    return torch.as_tensor(np.stack(vectors))


def initialize_active_learner(active_learner, y_train):

    indices_initial = random_initialization_stratified(y_train, 20)
    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial


if __name__ == '__main__':
    import argparse
    import logging
    logging.getLogger('small_text').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='An example that shows active learning '
                                                 'for multi-class text classification.')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='number of active learning iterations')

    args = parser.parse_args()

    main(num_iterations=args.num_iterations)
