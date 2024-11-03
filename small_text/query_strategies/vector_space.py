import numpy as np
import numpy.typing as npt

from typing import Union

from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

from small_text.classifiers import Classifier
from small_text.data import Dataset

from small_text.vector_indexes.base import VectorIndexFactory
from small_text.vector_indexes.hnsw import HNSWIndex

from small_text.query_strategies.strategies import EmbeddingBasedQueryStrategy


def _construct_graph(index, embeddings, max_distance=0.1, k=25):
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(list(range(embeddings.shape[0])))

    edges = []

    for i in range(embeddings.shape[0]):
        indices, distances = index.search(embeddings[i].reshape(1, -1), k=k, return_distance=True)

        for j, dist in zip(indices[0, 1:], distances[0, 1:]):
            if dist <= max_distance:
                edges.append((j, i))

    graph.add_edges_from(edges)
    return graph


class ProbCover(EmbeddingBasedQueryStrategy):
    """
    ProbCover [YDH+22]_ queries instances by trying to maximize probability coverage of an embedding space.
    For this, each labeled instance covers an area in the embedding space with a certain radius (`ball_radius`).
    The strategy tries to maximize the covered area, by selecting instances with a high-density neighborhood.

    .. versionadded:: 2.0.0
    """

    def __init__(self, vector_index_factory=VectorIndexFactory(HNSWIndex),
                 k=100, ball_radius=0.1):
        """
        Parameters
        ----------
        vector_index_factory : VectorIndexFactory, default=VectorIndexFactory(HNSWIndex)
            A factory that provides the vector index for nearest neighbor queries.
        k : int, default=100
            Number of nearest neighbors for nearest neighbor queries.
        ball_radius : float, default=0.1
            Radius of an embedding space ball that is given by a labeled instance at its center.
        """
        self.vector_index_factory = vector_index_factory
        self.k = k
        self.ball_radius = ball_radius

    def sample(self,
               clf: Classifier,
               dataset: Dataset,
               indices_unlabeled: npt.NDArray[np.uint],
               indices_labeled: npt.NDArray[np.uint],
               y: Union[npt.NDArray[np.uint], csr_matrix],
               n: int,
               embeddings: npt.NDArray[np.double],
               embeddings_proba: npt.NDArray[np.double] = None):

        embeddings = normalize(embeddings, axis=1)
        index = self.vector_index_factory.new()
        index.build(embeddings)

        # The paper explicitly mentions this part being implemented with sparse structures.
        # For now, I prefer the readability of this solution, but this might be changed in the future.
        graph = _construct_graph(index, embeddings, max_distance=self.ball_radius, k=self.k)

        indices_queried = []

        for i in indices_labeled:
            for node in list(graph.adj[i]):
                graph.remove_edge(node, i)

        for _ in range(n):
            index, _ = sorted(graph.degree(), key=lambda x: -x[1])[0]
            for node in list(graph.adj[index]):
                graph.remove_edge(node, index)
            indices_queried.append(index)

        return np.array(indices_queried)

    def __str__(self):
        # TODO: vector index factores dont have __str__ methods yet
        return (f'ProbCover(vector_index_factory={self.vector_index_factory.__class__.__name__}, '
                f'ball_radius={self.ball_radius}, ' \
                f'k={self.k})')
