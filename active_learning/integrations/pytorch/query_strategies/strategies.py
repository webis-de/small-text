import numpy as np

from sklearn.cluster import kmeans_plusplus

from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError
from active_learning.query_strategies import QueryStrategy
from active_learning.utils.context import build_pbar_context
from active_learning.utils.data import list_length

try:
    import torch
    import torch.nn.functional as F

    from active_learning.integrations.pytorch.utils.misc import assert_layer_exists, default_tensor_type
    from active_learning.integrations.pytorch.classifiers.kimcnn import kimcnn_collate_fn as default_collate_fn
    from active_learning.integrations.pytorch.utils.data import dataloader
except ImportError as e:
    raise PytorchNotFoundError('Could not import pytorch')


class ExpectedGradientLength(QueryStrategy):
    """Selects instances by expected gradient length [Set07]_.

    References
    ----------
    .. [Set07] Burr Settles, Mark Craven, and Soumya Ray. 2007.
       Multiple-instance active learning.
       In Proceedings of the 20th International Conference on Neural Information Processing Systems (NIPS’07).
       Curran Associates Inc., Red Hook, 1289–1296.
    """
    def __init__(self, num_classes, batch_size=50, device='cuda', pbar='tqdm'):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = device
        self.pbar = pbar

        self.scores_ = None

    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10, pbar=None):
        self._validate_query_input(x_indices_unlabeled, n)

        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)

        collate_fn = clf._create_collate_fn()
        dataset_iter = dataloader(x, batch_size=self.batch_size, collate_fn=collate_fn,
                                  train=False)

        clf.model.eval()
        clf.model.to(self.device)

        gradient_lengths = self.initialize_gradient_lengths_array(list_length(x))
        pbar_context = build_pbar_context('tqdm', tqdm_kwargs={'total': list_length(x)})

        offset = 0
        with default_tensor_type(torch.FloatTensor), pbar_context as pbar:
            for i, (x, _) in enumerate(dataset_iter):
                self.compute_gradient_lengths(clf, criterion, gradient_lengths, offset, x)

                batch_len = x.size(0)
                offset += batch_len

                if pbar is not None:
                    pbar.update(batch_len)

        return self.finalize_results(n, x_indices_unlabeled, gradient_lengths)

    def initialize_gradient_lengths_array(self, dim):
        return np.zeros(dim, dtype=np.float64)

    def finalize_results(self, n, x_indices_unlabeled, gradient_lengths):
        self.scores_ = gradient_lengths
        indices = np.argpartition(-gradient_lengths[x_indices_unlabeled], n)[:n]
        return np.array([x_indices_unlabeled[i] for i in indices])

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


class ExpectedGradientLengthMaxWord(ExpectedGradientLength):
    """Selects instances using the EGL-word model [ZLW17]_.

    The EGL-word model works as follows:

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

    References
    ----------
    .. [ZLW17] Ye Zhang, Matthew Lease, and Byron C. Wallace. 2017.
       Active discriminative text representation learning.
       In Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence (AAAI’17).
       AAAI Press, 3386–3392.
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

        # tensor that contains the unique word is for each item in the batch
        self._words = None

    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10, pbar=None):

        assert_layer_exists(clf.model, self.layer_name)
        return super().query(clf, x, x_indices_unlabeled, x_indices_labeled, y, n=n, pbar=pbar)

    def compute_gradient_lengths(self, clf, criterion, gradient_lengths, offset, x):

        self._words = torch.unique(x, dim=1).to(self.device)

        # TODO: assert layer_name is embedding layer
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

                # difference to ExpectedGradientLength: compute_gradients takes texts as argument
                self.compute_gradient_length(clf, x, sm, gradients, j, k)

        self.aggregate_gradient_lengths_batch(batch_len, gradient_lengths, gradients, offset)

    def compute_gradient_length(self, clf, text, sm, gradients, j, k):
        modules = dict({name: module for name, module in clf.model.named_modules()})

        params = list(modules[self.layer_name].parameters())
        assert len(params) == 1
        assert params[0].requires_grad is True
        params = params[0].grad

        word_indices = self._words[k]
        # <unk> and <pad> are currently not omitted
        # TODO: magic number 2(= len([<unk>, <pad>])
        # word_indices = word_indices[torch.gt(word_indices, 1)]
        params = params.index_select(index=word_indices, dim=0)

        norms = torch.norm(params, p=2, dim=1)
        max_norm = norms.max()

        gradients[j, k] = max_norm.item() * sm[k, j].item()

    def __str__(self):
        return 'ExpectedGradientLengthMaxWord()'


class ExpectedGradientLengthLayer(ExpectedGradientLength):

    def __init__(self, num_classes, layer_name, batch_size=50):

        super().__init__(num_classes, batch_size=batch_size)
        self.layer_name = layer_name

    def compute_gradient_length(self, clf, sm, gradients, j, k):

        assert_layer_exists(clf.model, self.layer_name)

        modules = dict({name: module for name, module in clf.model.named_modules()})
        params = [param.grad.flatten() for param in modules[self.layer_name].parameters()
                  if param.requires_grad]
        params = torch.cat(params)

        gradients[j, k] += torch.norm(params, 2)
        gradients[j, k] = gradients[j, k] * sm[k, j].item()

    def __str__(self):
        return 'ExpectedGradientLengthLayer()'


class BADGE(QueryStrategy):
    """
    Implements "Batch Active learning by Diverse Gradient Embedding" (BADGE) _[AZK20].

    References
    ----------
    .. [AZK+20] Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford and Alekh Agarwal. 2020.
                Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds.
                International Conference on Learning Representations 2020 (ICLR 2020).
    """
    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10, pbar=None,
              embed_kwargs=None):

        embed_kwargs = dict() if embed_kwargs is None else embed_kwargs
        embeddings = clf.embed(x[x_indices_unlabeled], pbar=pbar, **embed_kwargs)

        _, indices = kmeans_plusplus(embeddings,
                                     n,
                                     x_squared_norms=np.linalg.norm(embeddings, axis=1),
                                     random_state=np.random.RandomState())
        return np.array([x_indices_unlabeled[i] for i in indices])

    def __str__(self):
        return 'BADGE()'
