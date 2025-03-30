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
    random_initialization_stratified,
    list_to_csr
)

from examplecode.data.example_data_multilabel import (
    get_train_test
)
from examplecode.shared import evaluate_multi_label


def main(num_iterations=10):
    # Active learning parameters
    num_classes = 28
    model_args = SetFitModelArguments('sentence-transformers/paraphrase-mpnet-base-v2')
    # If GPU memory is a problem:
    # model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')

    clf_factory = SetFitClassificationFactory(model_args,
                                              num_classes,
                                              classification_kwargs={
                                                  'use_differentiable_head': True,
                                                  'device': 'cuda',
                                                  'max_seq_len': 512,
                                                  'mini_batch_size': 16,
                                                  'multi_label': True
                                              })

    query_strategy = BreakingTies()

    # Prepare some data
    train, test = get_train_test()

    train = TextDataset.from_arrays(train['text'], list_to_csr(train['labels'], (len(train), num_classes)), target_labels=np.arange(num_classes))
    test = TextDataset(test['text'], list_to_csr(test['labels'], (len(test), num_classes)), target_labels=np.arange(num_classes))

    # Active learner
    setfit_train_kwargs = {'show_progress_bar': False,
                           'num_epochs': (1, 1),
                           'end_to_end': True,
                           'body_learning_rate': (1e-5, 1e-5),
                           'head_learning_rate': 1e-5}
    active_learner = PoolBasedActiveLearner(clf_factory,
                                            query_strategy, train,
                                            fit_kwargs={'setfit_train_kwargs': setfit_train_kwargs})
    indices_labeled = initialize_active_learner(active_learner, train.y)

    # active_learner.save("test.pkl")

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

        print('Iteration #{:d} ({} samples)'.format(i, len(active_learner.indices_labeled)))
        evaluate_multi_label(active_learner, train[active_learner.indices_labeled], test)

def initialize_active_learner(active_learner, y_train):

    indices_initial = random_initialization_stratified(y_train, n_samples=28*20)
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
