import numpy as np
import numpy.typing as npt

from typing import Union

from scipy.sparse import csr_matrix
from scipy.special import softmax

from small_text.classifiers import Classifier
from small_text.data import Dataset
from small_text.query_strategies.strategies import DiscriminativeActiveLearning

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.query_strategies import (
    constraints,
    QueryStrategy,
    EmbeddingBasedQueryStrategy)
from small_text.utils.clustering import init_kmeans_plusplus_safe
from small_text.utils.context import build_pbar_context
from small_text.utils.data import list_length

try:
    import torch
    import torch.nn.functional as F  # noqa: N812

    from torch.amp import GradScaler  # pyright: ignore
    from torch.nn import BCEWithLogitsLoss
    from torch.nn.utils import clip_grad_norm_  # pyright: ignore

    from torch.optim import Adam

    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.pytorch.models.mlp import MLP

    from small_text.integrations.pytorch.utils.misc import _assert_layer_exists
    from small_text.integrations.pytorch.utils.data import dataloader
    from small_text.integrations.pytorch.utils.contextmanager import inference_mode
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


@constraints(classification_type='single-label')
class ExpectedGradientLength(QueryStrategy):
    """Selects instances by expected gradient length [Set07]_."""
    def __init__(self, num_classes, batch_size=50, device='cuda', pbar='tqdm'):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes.
        batch_size : int, default=50
            Batch size in which the query strategy scores the instances.
        device : str or torch.device, default=None
            Torch device on which the computation will be performed.
        pbar : 'tqdm' or None, default='tqdm'
            Displays a progress bar if 'tqdm' is passed.
        """
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = device
        self.pbar = pbar

        self.scores_ = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10, pbar=None):
        self._validate_query_input(indices_unlabeled, n)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)

        collate_fn = clf._create_collate_fn()
        dataset_iter = dataloader(dataset, batch_size=self.batch_size, collate_fn=collate_fn,
                                  train=False)

        clf.model.eval()
        clf.model.to(self.device)

        gradient_lengths = self.initialize_gradient_lengths_array(list_length(dataset))
        pbar_context = build_pbar_context('tqdm', tqdm_kwargs={'total': list_length(dataset)})

        offset = 0
        with pbar_context as pbar:
            for i, (dataset, *_) in enumerate(dataset_iter):
                self.compute_gradient_lengths(clf, criterion, gradient_lengths, offset, dataset)

                batch_len = dataset.size(0)
                offset += batch_len

                if pbar is not None:
                    pbar.update(batch_len)

        return self.finalize_results(n, indices_unlabeled, gradient_lengths)

    def initialize_gradient_lengths_array(self, dim):
        return np.zeros(dim, dtype=np.float64)

    def finalize_results(self, n, indices_unlabeled, gradient_lengths):
        self.scores_ = gradient_lengths
        indices = np.argpartition(-gradient_lengths[indices_unlabeled], n)[:n]
        return np.array([indices_unlabeled[i] for i in indices])

    def compute_gradient_lengths(self, clf, criterion, gradient_lengths, offset, x):

        batch_len = x.size(0)
        all_classes = torch.LongTensor([batch_len * [i]
                                        for i in range(self.num_classes)])
        if self.device is not None and self.device != 'cpu':
            all_classes = all_classes.to(self.device)

        gradients = self.initialize_gradients(batch_len)

        x = x.to(self.device, non_blocking=True)
        clf.model.zero_grad()

        self.compute_gradient_lengths_batch(clf, criterion, x, gradients, all_classes)
        self.aggregate_gradient_lengths_batch(batch_len, gradient_lengths, gradients, offset)

    def initialize_gradients(self, batch_len):
        return torch.zeros([self.num_classes, batch_len]).to(self.device, non_blocking=True)

    def compute_gradient_lengths_batch(self, clf, criterion, x, gradients, all_classes):

        batch_len = x.size(0)

        output = clf.model(x)
        with inference_mode():
            sm = F.softmax(output, dim=1)

        for j in range(self.num_classes):
            loss = criterion(output, all_classes[j])

            for k in range(batch_len):
                clf.model.zero_grad()
                loss[k].backward(retain_graph=True)

                self.compute_gradient_length(clf, sm, gradients, j, k)

    def compute_gradient_length(self, clf, sm, gradients, j, k):

        params = [param.grad.flatten() for param in clf.model.parameters()
                  if param.requires_grad]
        params = torch.cat(params)

        gradients[j, k] = torch.sqrt(params.pow(2)).sum()
        gradients[j, k] *= sm[k, j].item()

    def aggregate_gradient_lengths_batch(self, batch_len, gradient_lengths, gradients, offset):
        gradient_lengths[offset:offset + batch_len] = torch.sum(gradients, 0)\
            .to('cpu', non_blocking=True)

    def __str__(self):
        return 'ExpectedGradientLength()'


@constraints(classification_type='single-label')
class ExpectedGradientLengthMaxWord(ExpectedGradientLength):
    """Selects instances using the EGL-word strategy [ZLW17]_.

    The EGL-word strategy works as follows:

    1. For every instance and class the gradient norm is computed per word.
       The score for each (instance, class) pair is the norm of the word with the
       highest gradient norm value.
    2. These scores are then summed up over all classes.
       The result is one score per instance.

    Finally, the instances are selected by maximum score.

    Notes
    -----
    - An embedding layer is required for this strategy.
    - This strategy was designed for the `KimCNN` model and might not work for other models
      even if they posses an embedding layer.
    """
    def __init__(self, num_classes, layer_name, batch_size=50, device='cuda'):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes.
        layer_name : str
            Name of the embedding layer.
        batch_size : int
            Batch size.
        device : str or torch.device
            Torch device on which the computation will be performed.
        """

        super().__init__(num_classes, batch_size=batch_size, device=device)

        self.layer_name = layer_name

        # tensor of unique word ids
        self._words = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10, pbar=None):

        if clf.model is None:
            raise ValueError('Initial model must be trained!')

        _assert_layer_exists(clf.model, self.layer_name)
        modules = dict(clf.model.named_modules())
        if not isinstance(modules[self.layer_name], torch.nn.Embedding):
            raise ValueError(f'Given parameter (layer_name={self.layer_name}) '
                             f'is not an embedding layer.')

        return super().query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n, pbar=pbar)

    def compute_gradient_lengths(self, clf, criterion, gradient_lengths, offset, x):

        self._words = torch.unique(x, dim=1).to(self.device)

        batch_len = x.size(0)
        all_classes = torch.cuda.LongTensor([batch_len * [i]
                                             for i in range(self.num_classes)])
        gradients = self.initialize_gradients(batch_len)

        x = x.to(self.device, non_blocking=True)
        clf.model.zero_grad()

        output = clf.model(x)
        with inference_mode():
            sm = F.softmax(output, dim=1)

        for j in range(self.num_classes):
            loss = criterion(output, all_classes[j])

            for k in range(batch_len):
                clf.model.zero_grad()
                loss[k].backward(retain_graph=True)

                # Contrary to ExpectedGradientLength, compute_gradients operates on text
                self.compute_gradient_length(clf, x, sm, gradients, j, k)

        self.aggregate_gradient_lengths_batch(batch_len, gradient_lengths, gradients, offset)

    def compute_gradient_length(self, clf, text, sm, gradients, j, k):
        modules = dict({name: module for name, module in clf.model.named_modules()})

        params = list(modules[self.layer_name].parameters())
        assert len(params) == 1
        assert params[0].requires_grad is True
        params = params[0].grad

        word_indices = self._words[k]
        # special tokens (such as <unk> and <pad>) are currently not omitted
        params = params.index_select(index=word_indices, dim=0)

        norms = torch.norm(params, p=2, dim=1)
        max_norm = norms.max()

        gradients[j, k] = max_norm.item() * sm[k, j].item()

    def __str__(self):
        return 'ExpectedGradientLengthMaxWord()'


@constraints(classification_type='single-label')
class ExpectedGradientLengthLayer(ExpectedGradientLength):
    """An EGL variant that is restricted to the gradients of a single layer.

    This is a generalized version of the EGL-sm strategy [ZLW17]_, but instead of being
    restricted to the last layer it operates on the layer name passed to the constructor.
    """

    def __init__(self, num_classes, layer_name, batch_size=50):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes.
        layer_name : str
            Name of the target layer.
        batch_size : int, default=50
            Batch size in which the query strategy scores the instances.
        """
        super().__init__(num_classes, batch_size=batch_size)
        self.layer_name = layer_name

    def compute_gradient_length(self, clf, sm, gradients, j, k):

        _assert_layer_exists(clf.model, self.layer_name)

        modules = dict({name: module for name, module in clf.model.named_modules()})
        params = [param.grad.flatten() for param in modules[self.layer_name].parameters()
                  if param.requires_grad]
        params = torch.cat(params)

        gradients[j, k] += torch.norm(params, 2)
        gradients[j, k] = gradients[j, k] * sm[k, j].item()

    def __str__(self):
        return 'ExpectedGradientLengthLayer()'


@constraints(classification_type='single-label')
class BADGE(EmbeddingBasedQueryStrategy):
    """Implements "Batch Active learning by Diverse Gradient Embedding" (BADGE) [AZK+20]_.
    """
    def __init__(self, num_classes):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes.
        """
        self.num_classes = num_classes

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):

        if embeddings_proba is None:
            proba = clf.predict_proba(dataset[indices_unlabeled])
            embeddings = self.get_badge_embeddings(embeddings[indices_unlabeled],
                                                   proba)
        else:
            embeddings = self.get_badge_embeddings(embeddings[indices_unlabeled],
                                                   embeddings_proba[indices_unlabeled])

        _, indices = init_kmeans_plusplus_safe(embeddings,
                                               n,
                                               x_squared_norms=np.linalg.norm(embeddings, axis=1),
                                               random_state=np.random.RandomState())
        return indices

    def get_badge_embeddings(self, embeddings, proba):

        proba_argmax = np.argmax(proba, axis=1)
        scale = -1 * proba
        scale[proba_argmax] = -1 * proba[proba_argmax]

        if self.num_classes > 2:
            embedding_size = embeddings.shape[1]
            badge_embeddings = np.zeros((embeddings.shape[0], embedding_size * self.num_classes))
            for c in range(self.num_classes):
                badge_embeddings[:, c * embedding_size:(c + 1) * embedding_size] = (
                            scale[:, c] * np.copy(embeddings).T).T
        else:
            badge_embeddings = embeddings

        return badge_embeddings

    def __str__(self):
        return f'BADGE(num_classes={self.num_classes})'


def _discr_repr_learning_collate_fn(batch):
    return torch.Tensor(np.vstack([x for x, _ in batch])), torch.Tensor(np.array([y for _, y in batch])).long()


class DiscriminativeRepresentationLearning(QueryStrategy):
    """Discriminative Active Learning [GS19]_ learns to differentiate between the labeled and
    unlabeled pool and selects the instances that are most likely to belong to the unlabeled pool.

    This implementation uses embeddings as input representation to learn the discriminative binary problem.

    Note
    ----
    This is a variant of :py:class:`DiscriminativeActiveLearning` which is not only more efficient but is also
    reported to perform best in the blog post linked below. The default configuration is intended to adhere
    to wherever possible (except for the different setting which was image classification in the original publication.)

    See Also
    --------
    * `Blog post "Discriminative Active Learning" <BLOGPOST_DAL_>`__
        A detailed and highly informative blog post on Discriminative Active Learning in which
        the original author Daniel Gissin elaborates on the method.
    * `Original implementation <https://github.com/dsgissin/DiscriminativeActiveLearning>`__

    .. _BLOGPOST_DAL: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/DAL.html
    """

    def __init__(self,
                 num_iterations: int = 10,
                 selection: str = 'stochastic',
                 temperature: float = 0.01,
                 unlabeled_factor: int = 10,
                 mini_batch_size: int = 32,
                 device='cuda',
                 amp_args=None,
                 embed_kwargs: dict = {},
                 train_kwargs: dict = {},
                 pbar='tqdm'):
        """
        Parameters
        ----------
        num_iterations : int, default=10
            Number of iterations for the discriminative training.
        selection : {'stochastic', 'greedy'}, default='stochastic'
            Determines how the instances are selected. The option `stochastic` draws from the
            probability distribution over the current unlabeled instances that is given by the confidence estimate
            (`predict_proba()`) for the discriminative unlabeled class. The `greedy` selects n instances that are
            most likely to belong to the "unlabeled" class.
        temperature : float, default=1.0
            Temperature for the stochastic sampling (i.e., only applicable if `selection='stochastic'`).
            Lower values push the sampling distribution towards the one-hot categorical distribution, and higher
            values towards a uniform distribution [JGP+17]_.
        unlabeled_factor : int, default=10
            The ratio of "unlabeled pool" instances to "labeled pool" instances in the
            discriminative training.
        mini_batch_size : int, default=32
            Size of mini batches during training.
        device : str or torch.device, default='cuda'
            Torch device on which the computation will be performed.
        amp_args : AMPArguments or None, default=None
            Configures the use of Automatic Mixed Precision (AMP).

            .. seealso:: :py:class:`~small_text.integrations.pytorch.classifiers.base.AMPArguments`

        embed_kwargs : dict, default={}
            Keyword arguments that will be passed to the embed() method.
        train_kwargs : dict, default={}
            Keyword arguments with parameters for the training process within this method.

            Possible arguments:

            - `num_epochs` (int, default=4): Number of training epochs.
            - `lr` (float, default=2e-5): Learning rate.
            - `clip_grad_norm` (float, default=1): Gradients are clipped when their norm exceeds this value.

        pbar : 'tqdm' or None, default='tqdm'
            Displays a progress bar if 'tqdm' is passed.
        """
        self.num_iterations = num_iterations

        if selection not in set(['stochastic', 'greedy']):
            raise ValueError(f'Invalid selection strategy: {selection}')
        self.selection = selection

        if temperature <= 0:
            raise ValueError('Invalid temperature: temperature must be greater zero')
        self.temperature = temperature

        self.unlabeled_factor = unlabeled_factor
        self.device = device
        self._amp_args = amp_args
        self.pbar = pbar

        self.embed_kwargs = embed_kwargs
        self.train_kwargs = train_kwargs

        self.mini_batch_size = mini_batch_size

        self.clf_ = None

    def query(self,
              clf: Classifier,
              dataset: Dataset,
              indices_unlabeled: npt.NDArray[np.uint],
              indices_labeled: npt.NDArray[np.uint],
              y: Union[npt.NDArray[np.uint], csr_matrix],
              n: int = 10) -> np.ndarray:
        self._validate_query_input(indices_unlabeled, n)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        query_sizes = DiscriminativeActiveLearning._get_query_sizes(self.num_iterations, n)
        return self._discriminative_active_learning(clf, dataset, indices_unlabeled, indices_labeled,
                                                    query_sizes)

    @property
    def amp_args(self):
        if self._amp_args is None:
            device_type = 'cpu' if self.device is None else self.device
            amp_args = AMPArguments(device_type=device_type, dtype=torch.bfloat16)
        else:
            amp_args = AMPArguments(use_amp=self._amp_args.use_amp,
                                    device_type=self._amp_args.device_type,
                                    dtype=self._amp_args.dtype)
        if self.device is None or self.device == 'cpu':
            amp_args.use_amp = False
        return amp_args

    def _discriminative_active_learning(self, clf, dataset, indices_unlabeled, indices_labeled,
                                        query_sizes):

        indices = np.array([], dtype=indices_labeled.dtype)

        indices_unlabeled_copy = np.copy(indices_unlabeled)
        indices_labeled_copy = np.copy(indices_labeled)

        embeddings = clf.embed(dataset)

        input_size = embeddings.shape[1]
        hidden_size = self._get_hidden_size(clf.model)

        with build_pbar_context(len(query_sizes)) as pbar:
            for q in query_sizes:
                discr_model = MLP(input_size, hidden_size, 2).to(self.device)
                discr_model = discr_model.to(self.device)

                indices_most_confident = self._train_and_get_most_confident(discr_model,
                                                                            embeddings,
                                                                            indices_unlabeled_copy,
                                                                            indices_labeled_copy,
                                                                            q)

                indices = np.append(indices, indices_unlabeled_copy[indices_most_confident])
                indices_labeled_copy = np.append(indices_labeled_copy,
                                                 indices_unlabeled_copy[indices_most_confident])
                indices_unlabeled_copy = np.delete(indices_unlabeled_copy, indices_most_confident)
                pbar.update(1)

                del discr_model

        return indices

    def _get_hidden_size(self, model):
        if hasattr(model, 'config'):
            return model.config.hidden_size
        elif hasattr(model, 'model_body'):
            vec = model.model_body.encode('')
            return vec.shape[0]
        elif hasattr(model, 'convs'):
            return model.embedding.embedding_dim

        raise ValueError(f'Incompatible model class {type(model).__name__}')

    def _train_and_get_most_confident(self, discr_model, embeddings, indices_unlabeled, indices_labeled, q):
        discr_model.train()

        num_unlabeled = min(indices_labeled.shape[0] * self.unlabeled_factor,
                            indices_unlabeled.shape[0])

        indices_unlabeled_sub = np.random.choice(indices_unlabeled,
                                                 num_unlabeled,
                                                 replace=False)

        y = np.array([DiscriminativeActiveLearning.LABEL_UNLABELED_POOL] * indices_unlabeled_sub.shape[0] +
                     [DiscriminativeActiveLearning.LABEL_LABELED_POOL] * indices_labeled.shape[0])

        self._train(discr_model, embeddings, y)

        # return instances which most likely belong to the "unlabeled" class (higher is better)
        if self.selection == 'stochastic':
            proba = self._predict(discr_model, embeddings[indices_unlabeled_sub], False)
            proba = np.log(softmax(proba / self.temperature))

            indices = []
            for j in range(q):
                proba_new = proba + np.random.gumbel(size=proba.shape[0])
                proba_new[indices] = -np.inf
                indices.append(np.argmax(proba_new))

            return indices
        else:
            proba = self._predict(discr_model, embeddings[indices_unlabeled_sub], True)
            return np.argpartition(-proba, q)[:q]

    def _train(self, discr_model, x, y):
        base_lr = self.train_kwargs.get('lr', 2e-5)
        num_epochs = self.train_kwargs.get('num_epochs', 4)
        clip_grad_norm = self.train_kwargs.get('clip_grad_norm', 1)

        optimizer = Adam(discr_model.parameters(), lr=base_lr, eps=1e-8, weight_decay=0)
        scaler = GradScaler(enabled=self.amp_args.use_amp)

        criterion = BCEWithLogitsLoss()

        data = list(zip(x, y))

        with torch.autocast(enabled=self.amp_args.use_amp, device_type=self.amp_args.device_type,
                            dtype=self.amp_args.dtype):
            for _ in range(num_epochs):
                dataset_iter = dataloader(data, batch_size=self.mini_batch_size, train=True,
                                          collate_fn=_discr_repr_learning_collate_fn)
                for representation, cls in dataset_iter:
                    representation = representation.to(self.device)
                    cls = cls.to(self.device)

                    optimizer.zero_grad()

                    output = discr_model(representation)
                    target = F.one_hot(cls, 2).float()

                    loss = criterion(output, target)
                    loss = loss.mean()

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    clip_grad_norm_(discr_model.parameters(), clip_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()

    def _predict(self, discr_model, representations, return_probas):
        discr_model.eval()

        data = list(zip(representations, np.zeros_like(representations)))

        predictions = np.empty((representations.shape[0], ), dtype=float)
        offset = 0

        with inference_mode():
            with torch.autocast(device_type=self.amp_args.device_type, dtype=torch.bfloat16,
                                enabled=self.amp_args.use_amp):
                dataset_iter = dataloader(data, batch_size=self.mini_batch_size, train=False,
                                          collate_fn=_discr_repr_learning_collate_fn)
                for representation, _ in dataset_iter:
                    batch_size = representation.shape[0]

                    representation = representation.to(self.device)
                    output = discr_model(representation)

                    if return_probas:
                        prediction = F.softmax(output, dim=1)
                    else:
                        prediction = output
                    prediction = prediction[:, DiscriminativeActiveLearning.LABEL_UNLABELED_POOL]
                    prediction = prediction.float().to('cpu').numpy()

                    predictions[offset:offset+batch_size] = prediction
                    offset += batch_size

        return predictions

    def __str__(self):
        return f'DiscriminativeRepresentationLearning(' \
               f'num_iterations={self.num_iterations}, selection={self.selection}, temperature={self.temperature}, ' \
               f'unlabeled_factor={self.unlabeled_factor}, mini_batch_size={self.mini_batch_size}, ' \
               f'device={self.device})'
