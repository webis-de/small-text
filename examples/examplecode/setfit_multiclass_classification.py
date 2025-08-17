"""Example of a setfit-based active learning multi-class text classification.
"""
import numpy as np

from small_text import (
    EmptyPoolException,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    BreakingTies,
    SetFitClassificationFactory,
    SetFitModelArguments,
    TextDataset,
    random_initialization_balanced
)

from examplecode.data.corpus_twenty_news import get_twenty_newsgroups_corpus
from examplecode.shared import evaluate


TWENTY_NEWS_SUBCATEGORIES = ['rec.sport.baseball', 'sci.med', 'rec.autos']


def main(num_iterations=10):
    # Active learning parameters
    num_classes = len(TWENTY_NEWS_SUBCATEGORIES)
    model_args = SetFitModelArguments('sentence-transformers/paraphrase-mpnet-base-v2',
                                      max_length=64,
                                      mini_batch_size=8,
                                      show_progress_bar=False)
    # If GPU memory is a problem:
    # model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')

    clf_factory = SetFitClassificationFactory(model_args,
                                              num_classes,
                                              classification_kwargs={
                                                  'device': 'cuda'
                                              })

    query_strategy = BreakingTies()

    # Prepare some data
    train, test = get_twenty_newsgroups_corpus(categories=TWENTY_NEWS_SUBCATEGORIES)
    train.data = [txt for txt in train.data]

    train = TextDataset.from_arrays(train.data, train.target, target_labels=np.arange(num_classes))
    test = TextDataset(test.data, test.target, target_labels=np.arange(num_classes))

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
        indices_queried = active_learner.query(num_samples=20)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print('Iteration #{:d} ({} samples)'.format(i, len(indices_labeled)))
        evaluate(active_learner, train[indices_labeled], test)


def initialize_active_learner(active_learner, y_train):

    indices_initial = random_initialization_balanced(y_train)
    active_learner.initialize(indices_initial)

    return indices_initial


if __name__ == '__main__':
    import argparse
    import logging
    logging.getLogger('small_text').setLevel(logging.INFO)

    for logger_name in ['setfit.modeling', 'setfit.trainer']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description='An example that shows active learning '
                                                 'for multi-class text classification '
                                                 'using a setfit classifier.')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='number of active learning iterations')

    args = parser.parse_args()

    main(num_iterations=args.num_iterations)
