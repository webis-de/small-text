import types
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from small_text.base import check_optional_dependency
from small_text.classifiers.classification import Classifier, EmbeddingMixin
from small_text.exceptions import UnsupportedOperationException
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

from small_text.utils.classification import (
    _check_classifier_dataset_consistency,
    empty_result,
    _multi_label_list_to_multi_hot,
    prediction_result
)
from small_text.utils.context import build_pbar_context
from small_text.utils.labels import csr_to_list
from small_text.integrations.transformers.classifiers.base import (
    ModelLoadingStrategy
)

from small_text.integrations.transformers.utils.setfit import (
    _check_model_kwargs,
    _check_trainer_kwargs,
    _check_train_kwargs,
    _truncate_texts
)

try:
    import torch

    from datasets import Dataset
    from setfit import SetFitModel, SetFitTrainer, TrainingArguments

    from small_text.integrations.pytorch.classifiers.base import AMPArguments

    from small_text.integrations.pytorch.utils.contextmanager import inference_mode
    from small_text.integrations.pytorch.utils.misc import _compile_if_possible, enable_dropout
    from small_text.integrations.transformers.utils.classification import (
        _get_arguments_for_from_pretrained_model
    )
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


class SetFitModelArguments(object):
    """Model arguments for :py:class:`SetFitClassification`.

    .. versionadded:: 1.2.0
    """

    def __init__(self,
                 sentence_transformer_model: str,
                 model_kwargs={},
                 trainer_kwargs={},
                 model_loading_strategy: ModelLoadingStrategy = ModelLoadingStrategy.DEFAULT,
                 compile_model: bool = False):
        """
        Parameters
        ----------
        sentence_transformer_model : str
            Name of a sentence transformer model.
        model_kwargs : dict, default={}
            Keyword arguments used for the SetFit model. The keyword `use_differentiable_head` is
            excluded and managed by this class. The other keywords are directly passed to
            `SetFitModel.from_pretrained()`. Additional kwargs that will be passed into
            `SetFitModel.from_pretrained()`. Arguments that are managed by small-text
            (such as the model name given by `model`) are excluded.

            .. seealso::

                `SetFitModel.from_pretrained()
                <https://huggingface.co/docs/setfit/en/reference/main#setfit.SetFitModel.from_pretrained>`_
                in the SetFit documentation.
        trainer_kwargs : dict, default={}
            Keyword arguments used for the SetFit model. The keyword `batch_size` is excluded and
            is instead controlled by the keyword `mini_batch_size` of this class. The other
            keywords are directly passed to `SetFitTrainer.__init__()`.

            .. seealso:: `Trainer
                         <https://huggingface.co/docs/setfit/en/reference/trainer>`_
                         in the SetFit documentation.
        model_loading_strategy: ModelLoadingStrategy, default=ModelLoadingStrategy.DEFAULT
            Specifies if there should be attempts to download the model or if only local
            files should be used.
        compile_model : bool, default=False
            Compiles the model (using `torch.compile`) if `True` and provided that
            the PyTorch version greater or equal 2.0.0.

            .. versionadded:: 2.0.0
        """
        self.sentence_transformer_model = sentence_transformer_model
        self.model_kwargs = _check_model_kwargs(model_kwargs)
        self.trainer_kwargs = _check_trainer_kwargs(trainer_kwargs)
        self.model_loading_strategy = model_loading_strategy
        self.compile_model = compile_model


class SetFitClassificationEmbeddingMixin(EmbeddingMixin):
    """
    .. versionadded:: 1.2.0
    """

    def embed(self, data_set, return_proba=False, pbar='tqdm'):
        """Embeds each sample in the given `data_set`.

        The embedding is created by using the underlying sentence transformer model.

        Parameters
        ----------
        data_set : TextDataset
            The dataset for which embeddings (and class probabilities) will be computed.
        return_proba : bool
            Also return the class probabilities for `data_set`.
        pbar : 'tqdm' or None, default='tqdm'
            Displays a progress bar if 'tqdm' is passed.

        Returns
        -------
        embeddings : np.ndarray
            Embeddings in the shape (N, hidden_layer_dimensionality).
        proba : np.ndarray
            Class probabilities for `data_set` (only if `return_predictions` is `True`).
        """

        if self.model is None:
            raise ValueError('Model is not trained. Please call fit() first.')

        if self.use_differentiable_head is False:
            try:
                check_is_fitted(self.model.model_head)
            except NotFittedError:
                raise ValueError('Model is initialized but not trained. Please call fit() first.')

        data_set = _truncate_texts(self.model, self.max_length, data_set)[0]

        embeddings = []
        predictions = []

        num_batches = int(np.ceil(len(data_set) / self.mini_batch_size))
        with build_pbar_context(pbar, tqdm_kwargs={'total': len(data_set)}) as pbar:
            with torch.autocast(device_type=self.amp_args.device_type, dtype=self.amp_args.dtype,
                                enabled=self.amp_args.use_amp):
                for batch in np.array_split(data_set.x, num_batches, axis=0):

                    batch_embeddings, probas = self._create_embeddings(batch)
                    pbar.update(batch_embeddings.shape[0])
                    embeddings.extend(batch_embeddings.tolist())
                    if return_proba:
                        predictions.extend(probas.tolist())

        if return_proba:
            return np.array(embeddings), np.array(predictions)

        return np.array(embeddings)

    def _create_embeddings(self, texts):

        if self.use_differentiable_head:
            embeddings = self.model.model_body.encode(texts, convert_to_tensor=True, device=self.device)
            proba = self.model.model_head.predict_proba(embeddings)
        else:
            embeddings = self.model.model_body.encode(texts, device=self.device)
            proba = self.model.model_head.predict_proba(embeddings)

        return embeddings, proba


class SetFitClassification(SetFitClassificationEmbeddingMixin, Classifier):
    """A classifier that operates through Sentence Transformer Finetuning (SetFit, [TRE+22]_).

    This class is a wrapper which encapsulates the
    `Hugging Face SetFit implementation <https://github.com/huggingface/setfit>_` .

    .. note ::
       This strategy requires the optional dependency `setfit`.

    .. versionadded:: 1.2.0
    """

    def __init__(self, setfit_model_args, num_classes, multi_label=False, max_length=512,
                 use_differentiable_head=False, mini_batch_size=32, amp_args=None, device=None):
        """
        sentence_transformer_model : SetFitModelArguments
            Settings for the sentence transformer model to be used.
        num_classes : int
            Number of classes.
        multi_label : bool, default=False
            If `False`, the classes are mutually exclusive, i.e. the prediction step results in
            exactly one predicted label per instance.
        max_length : int, default=512
            Maximum number of tokens. Tokens beyond that threshold will be discarded.
        use_differentiable_head : bool
            Uses a differentiable head instead of a logistic regression for the classification head.
            Corresponds to the keyword argument with the same name in
            `SetFitModel.from_pretrained()`.
        amp_args : AMPArguments, default=None
            Configures the use of Automatic Mixed Precision (AMP). Only affects the training.

            .. seealso:: :py:class:`~small_text.integrations.pytorch.classifiers.base.AMPArguments`
            .. versionadded:: 2.0.0

        device : str or torch.device, default=None
            Torch device on which the computation will be performed.
        """
        check_optional_dependency('setfit')

        self.setfit_model_args = setfit_model_args
        self.num_classes = num_classes
        self.multi_label = multi_label

        self.model = None

        self.max_length = max_length
        self.use_differentiable_head = use_differentiable_head
        self.mini_batch_size = mini_batch_size
        self._amp_args = amp_args
        self.device = device

    def fit(self, train_set, validation_set=None, setfit_train_kwargs=dict()):
        """Trains the model using the given train set.

        Parameters
        ----------
        train_set : TextDataset
            A dataset used for training the model.
        validation_set : TextDataset or None, default None
            A dataset used for validation during training.
        setfit_train_kwargs : dict
            Additional keyword arguments that are passed to `SetFitTrainer.train()`

        Returns
        -------
        self : SetFitClassification
            Returns the current classifier with a fitted model.
        """
        _check_classifier_dataset_consistency(self, train_set, dataset_name_in_error='training')
        _check_classifier_dataset_consistency(self, validation_set, dataset_name_in_error='validation')

        setfit_train_kwargs = _check_train_kwargs(setfit_train_kwargs)
        if self.model is None:
            self.model = self.initialize()

        if validation_set is None:
            train_set = _truncate_texts(self.model, self.max_length, train_set)[0]
        else:
            train_set, validation_set = _truncate_texts(self.model, self.max_length, train_set, validation_set)

        x_valid = validation_set.x if validation_set is not None else None
        y_valid = validation_set.y if validation_set is not None else None

        if self.multi_label:
            y_valid = _multi_label_list_to_multi_hot(csr_to_list(y_valid), self.num_classes) \
                if y_valid is not None else None
            y_train = _multi_label_list_to_multi_hot(csr_to_list(train_set.y), self.num_classes)

            sub_train, sub_valid = self._get_train_and_valid_sets(train_set.x,
                                                                  y_train,
                                                                  x_valid,
                                                                  y_valid)
        else:
            y_valid = y_valid.tolist() if isinstance(y_valid, np.ndarray) else y_valid
            sub_train, sub_valid = self._get_train_and_valid_sets(train_set.x,
                                                                  train_set.y,
                                                                  x_valid,
                                                                  y_valid)

        self.model.model_body.to(self.device)
        return self._fit(sub_train, sub_valid, setfit_train_kwargs)

    def _get_train_and_valid_sets(self, x_train, y_train, x_valid, y_valid):
        sub_train = Dataset.from_dict({'text': x_train, 'label': y_train})
        if x_valid is not None:
            sub_valid = Dataset.from_dict({'text': x_valid, 'label': y_valid})
        else:
            if self.use_differentiable_head:
                raise NotImplementedError
            else:
                sub_valid = None
        return sub_train, sub_valid

    def _fit(self, sub_train, sub_valid, setfit_train_kwargs):
        seed = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32).item()
        trainer = SetFitTrainer(
            self.model,
            sub_train,
            eval_dataset=sub_valid,
            batch_size=self.mini_batch_size,
            use_amp=self.amp_args.use_amp,
            seed=seed,
            **self.setfit_model_args.trainer_kwargs
        )
        # TODO:
        train_args = TrainingArguments(num_epochs=(1, 5),
                                       end_to_end=False,
                                       max_length=self.max_length,
                                       head_learning_rate=2e-5)
        trainer.train(args=train_args, **setfit_train_kwargs)
        return self

    def initialize(self):
        from_pretrained_options = _get_arguments_for_from_pretrained_model(
            self.setfit_model_args.model_loading_strategy
        )
        model_kwargs = self.setfit_model_args.model_kwargs.copy()
        if self.use_differentiable_head:
            model_kwargs['head_params'] = {'out_features': self.num_classes, 'multitarget': True}
        else:
            if self.multi_label and 'multi_target_strategy' not in model_kwargs:
                model_kwargs['multi_target_strategy'] = 'one-vs-rest'

        self.model = SetFitModel.from_pretrained(
            self.setfit_model_args.sentence_transformer_model,
            use_differentiable_head=self.use_differentiable_head,
            force_download=from_pretrained_options.force_download,
            local_files_only=from_pretrained_options.local_files_only,
            **model_kwargs
        )
        self.model.model_body = _compile_if_possible(self.model.model_body, compile_model=self.setfit_model_args.compile_model)
        return self.model

    def validate(self, _validation_set):
        if self.use_differentiable_head:
            raise NotImplementedError()
        else:
            raise UnsupportedOperationException(
                'validate() is not available when use_differentiable_head is set to False'
            )

    def predict(self, dataset, return_proba=False):
        """Predicts the labels for the given dataset.

        Parameters
        ----------
        dataset : TextDataset
            A dataset on whose instances predictions are made.
        return_proba : bool, default=False
            If True, additionally returns the confidence distribution over all classes.

        Returns
        -------
        predictions : np.ndarray[np.int32] or csr_matrix[np.int32]
            List of predictions if the classifier was fitted on single-label data,
            otherwise a sparse matrix of predictions.
        probas : np.ndarray[np.float32], optional
            List of probabilities (or confidence estimates) if `return_proba` is True.
        """
        if len(dataset) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=True,
                                return_proba=return_proba)

        proba = self.predict_proba(dataset)
        predictions = prediction_result(proba, self.multi_label, self.num_classes)

        if return_proba:
            return predictions, proba

        return predictions

    def predict_proba(self, dataset, dropout_sampling=1):
        """Predicts the label distributions.

        Parameters
        ----------
        dataset : TextDataset
            A dataset whose labels will be predicted.
        dropout_sampling : int, default=1
            If `dropout_sampling > 1` then all dropout modules will be enabled during prediction and
            multiple rounds of predictions will be sampled for each instance.

        Returns
        -------
        scores : np.ndarray
            Distribution of confidence scores over all classes of shape (num_samples, num_classes).
            If `dropout_sampling > 1` then the shape is (num_samples, dropout_sampling, num_classes).

        .. warning::
           This function is not thread-safe if `dropout_sampling > 1`, since the underlying model gets
           temporarily modified.
        """
        if len(dataset) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=False,
                                return_proba=True)
        dataset = _truncate_texts(self.model, self.max_length, dataset)[0]

        # if self.use_differentiable_head:
        #     raise NotImplementedError()

        with inference_mode():
            if dropout_sampling <= 1:
                return self._predict_proba(dataset)
            else:
                return self._predict_proba_dropout_sampling(dataset, dropout_samples=dropout_sampling)

    def _predict_proba(self, dataset):
        proba = np.empty((0, self.num_classes), dtype=float)

        num_batches = int(np.ceil(len(dataset) / self.mini_batch_size))
        for batch in np.array_split(dataset.x, num_batches, axis=0):
            proba_tmp = self.model.predict_proba(batch).cpu().detach().numpy()
            proba = np.append(proba, proba_tmp, axis=0)

        return proba

    def _predict_proba_dropout_sampling(self, dataset, dropout_samples=2):
        # this whole method be done much more efficiently but this solution works without modifying setfit's code

        self.model.model_body.train()
        model_body_eval = self.model.model_body.eval
        self.model.model_body.eval = types.MethodType(lambda x: x, self.model.model_body)

        proba = np.empty((0, dropout_samples, self.num_classes), dtype=float)
        proba[:, :, :] = np.inf

        with enable_dropout(self.model.model_body):
            num_batches = int(np.ceil(len(dataset) / self.mini_batch_size))
            for batch in np.array_split(dataset.x, num_batches, axis=0):
                samples = np.empty((dropout_samples, len(batch), self.num_classes), dtype=float)
                for i in range(dropout_samples):
                    proba_tmp = np.zeros((batch.shape[0], self.num_classes), dtype=float)
                    proba_tmp[:, self.model.model_head.classes_] = self.model.predict_proba(batch)
                    samples[i] = proba_tmp

                samples = np.swapaxes(samples, 0, 1)
                proba = np.append(proba, samples, axis=0)

        self.model.model_body.eval = model_body_eval

        return proba

    @property
    def amp_args(self):
        if self._amp_args is None:
            device_type = 'cpu' if self.model is None else self.model.model_body.device.type
            amp_args = AMPArguments(device_type=device_type, dtype=torch.bfloat16)
        else:
            amp_args = AMPArguments(use_amp=self._amp_args.use_amp,
                                    device_type=self._amp_args.device_type,
                                    dtype=self._amp_args.dtype)
        if self.model is None or self.model.model_body.device.type == 'cpu':
            amp_args.use_amp = False
        return amp_args

    def __del__(self):
        try:
            attrs = ['model']
            for attr in attrs:
                delattr(self, attr)
        except Exception:
            pass
