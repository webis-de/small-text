import numpy as np

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

    from small_text.integrations.pytorch.utils.misc import _assert_layer_exists
    from small_text.integrations.pytorch.utils.data import dataloader
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
        with torch.no_grad():
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
        with torch.no_grad():
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
