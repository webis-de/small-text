import unittest


class ImportTest(unittest.TestCase):

    def test_import_from_main_module(self):
        from small_text import ActiveLearner
        from small_text import AbstractPoolBasedActiveLearner
        from small_text import PoolBasedActiveLearner
        from small_text import LABEL_UNLABELED
        from small_text import LABEL_IGNORED
        from small_text import OPTIONAL_DEPENDENCIES
        from small_text import check_optional_dependency
        from small_text import ActiveLearnerException
        from small_text import ConstraintViolationError
        from small_text import LearnerNotInitializedException
        from small_text import MissingOptionalDependencyError
        from small_text import get_version

    def test_import_classifiers_module(self):
        from small_text import Classifier
        from small_text import SklearnClassifier
        from small_text import EmbeddingMixin
        from small_text import ConfidenceEnhancedLinearSVC
        from small_text import AbstractClassifierFactory
        from small_text import SklearnClassifierFactory

    def test_import_data_module(self):
        from small_text import check_size
        from small_text import check_dataset_and_labels
        from small_text import check_target_labels
        from small_text import Dataset
        from small_text import DatasetView
        from small_text import SklearnDatasetView
        from small_text import TextDatasetView
        from small_text import get_updated_target_labels
        from small_text import is_multi_label
        from small_text import select
        from small_text import SklearnDataset
        from small_text import TextDataset
        from small_text import split_data
        from small_text import balanced_sampling
        from small_text import multilabel_stratified_subsets_sampling
        from small_text import stratified_sampling
        from small_text import UnsupportedOperationException

    def test_import_initialization_module(self):
        from small_text import random_initialization
        from small_text import random_initialization_balanced
        from small_text import random_initialization_stratified

    def test_import_query_strategies_module(self):
        from small_text import constraints
        from small_text import ClassificationType
        from small_text import EmptyPoolException
        from small_text import QueryException
        from small_text import PoolExhaustedException
        from small_text import QueryStrategy
        from small_text import RandomSampling
        from small_text import ConfidenceBasedQueryStrategy
        from small_text import BreakingTies
        from small_text import LeastConfidence
        from small_text import PredictionEntropy
        from small_text import SubsamplingQueryStrategy
        from small_text import EmbeddingBasedQueryStrategy
        from small_text import EmbeddingKMeans
        from small_text import ContrastiveActiveLearning
        from small_text import DiscriminativeActiveLearning
        from small_text import SEALS
        from small_text import greedy_coreset
        from small_text import GreedyCoreset
        from small_text import lightweight_coreset
        from small_text import LightweightCoreset
        from small_text import BALD
        from small_text import CategoryVectorInconsistencyAndRanking

    def test_import_stopping_criteria_module(self):
        from small_text import ClassificationChange
        from small_text import DeltaFScore
        from small_text import StoppingCriterion
        from small_text import check_window_based_predictions
        from small_text import KappaAverage
        from small_text import OverallUncertainty

    def test_import_training_module(self):
        from small_text import EarlyStoppingHandler
        from small_text import NoopEarlyStopping
        from small_text import EarlyStopping
        from small_text import EarlyStoppingOrCondition
        from small_text import EarlyStoppingAndCondition
        from small_text import Metric
        from small_text import ModelSelectionResult
        from small_text import ModelSelectionManager
        from small_text import NoopModelSelection
        from small_text import ModelSelection

    def test_import_utils_module(self):
        from small_text import DeprecationError
        from small_text import ExperimentalWarning
        from small_text import split_data
        from small_text import prediction_result
        from small_text import empty_result
        from small_text import init_kmeans_plusplus_safe
        from small_text import build_pbar_context
        from small_text import NullProgressBar
        from small_text import check_training_data
        from small_text import list_length
        from small_text import get_num_labels
        from small_text import csr_to_list
        from small_text import list_to_csr
        from small_text import verbosity_logger
        from small_text import VerbosityLogger
        from small_text import VERBOSITY_QUIET
        from small_text import VERBOSITY_VERBOSE
        from small_text import VERBOSITY_MORE_VERBOSE
        from small_text import VERBOSITY_ALL
        from small_text import get_tmp_dir_base
        from small_text import TMP_DIR_VARIABLE
