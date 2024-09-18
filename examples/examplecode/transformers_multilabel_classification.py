"""Example of a transformer-based active learning multi-label text classification.

Note:
This examples requires the datasets library.
"""
from transformers import AutoTokenizer

from small_text import (
    ActiveLearnerException,
    EmptyPoolException,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    RandomSampling,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
    list_to_csr,
    random_initialization_stratified
)

from examplecode.data.example_data_multilabel import (
    get_train_test
)
from examplecode.data.example_data_transformers import preprocess_data
from examplecode.shared import evaluate_multi_label


TRANSFORMER_MODEL = TransformerModelArguments('distilroberta-base')


try:
    import datasets  # noqa: F401
except ImportError:
    raise ActiveLearnerException('This example requires the "datasets" library. '
                                 'Please install datasets to run this example.')


def main(num_iterations=10):
    # Active learning parameters
    num_classes = 28
    clf_factory = TransformerBasedClassificationFactory(TRANSFORMER_MODEL,
                                                        num_classes,
                                                        classification_kwargs={
                                                            'device': 'cuda',
                                                            'multi_label': True
                                                        })
    query_strategy = RandomSampling()

    # Prepare some data
    train, test = get_train_test()

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL.model, cache_dir='.cache/')
    train = preprocess_data(tokenizer,
                            train['text'],
                            list_to_csr(train['labels'], (len(train), num_classes)))

    test = preprocess_data(tokenizer,
                           test['text'],
                           list_to_csr(test['labels'], (len(test), num_classes)))

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
    # Perform 10 iterations of active learning...
    for i in range(num_iterations):
        # ...where each iteration consists of labelling 20 samples
        indices_queried = active_learner.query(num_samples=1000)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        print('Iteration #{:d} ({} samples)'.format(i, len(active_learner.indices_labeled)))
        evaluate_multi_label(active_learner, train[active_learner.indices_labeled], test)


def initialize_active_learner(active_learner, y_train):

    indices_initial = random_initialization_stratified(y_train, n_samples=2000)
    active_learner.initialize(indices_initial)

    return indices_initial


if __name__ == '__main__':
    import argparse
    import logging
    logging.getLogger('small_text').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='An example that shows active learning '
                                                 'for multi-class multi-label text classification '
                                                 'using transformers.')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='number of active learning iterations')

    args = parser.parse_args()

    main(num_iterations=args.num_iterations)
