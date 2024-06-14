import numpy as np

from small_text.classifiers import SklearnClassifier


class SklearnClassifierWithRandomEmbeddings(SklearnClassifier):

    def embed(self, dataset, embed_dim=5, pbar=None):
        _unused = pbar  # noqa:F841
        self.embeddings_ = np.random.rand(len(dataset), embed_dim)
        return self.embeddings_


class SklearnClassifierWithRandomEmbeddingsAndProba(SklearnClassifier):

    def embed(self, dataset, return_proba=False, embed_dim=5, pbar=None):
        self.embeddings_ = np.random.rand(len(dataset), embed_dim)
        _unused = pbar  # noqa:F841
        if return_proba:
            self.proba_ = np.random.rand(len(dataset))
            return self.embeddings_, self.proba_

        return self.embeddings_
