import numpy as np


class VectorIndexesTest(object):

    def get_vector_index(self):
        """
        Defines the vector index to be used for these test cases.
        """
        raise NotImplementedError

    def _get_random_data(self, n=20, d=4):
        return np.random.rand(n, d).astype(np.float32)

    def test_build(self):
        vector_index = self.get_vector_index()
        self.assertIsNone(vector_index.index)

        vectors = self._get_random_data()
        vector_index.build(vectors)
        self.assertIsNotNone(vector_index.index)

    def test_remove(self):
        vector_index = self.get_vector_index()
        vectors = self._get_random_data()
        vector_index.build(vectors)

        # before removal this does not raise an exception
        vector_index.search(vectors[[0], :], k=11)

        indices_to_remove = np.arange(10)
        vector_index.remove(indices_to_remove)
        with self.assertRaisesRegex(ValueError, 'Searching the vector index failed.'):
            vector_index.search(vectors[[0], :], k=11)

        indices = vector_index.search(vectors[[0], :], k=10)[0]
        removed_indices_in_result = set(indices.tolist()).intersection(set(indices_to_remove.tolist()))
        self.assertEqual(0, len(removed_indices_in_result))

    def test_search(self):
        vector_index = self.get_vector_index()
        vectors = self._get_random_data()
        vector_index.build(vectors)

        indices = vector_index.search(vectors[[0], :], k=10)
        valid_indices = np.arange(vectors.shape[0])
        self.assertTrue(np.isin(indices.flatten(), valid_indices).all())
        self.assertEqual(10, indices.ravel().shape[0])
        self.assertEqual(0, indices.ravel()[0].item())

    def test_search_with_ids(self):
        vector_index = self.get_vector_index()
        vectors = self._get_random_data()
        ids = np.random.choice(np.arange(1024), size=vectors.shape[0])
        vector_index.build(vectors, ids=ids)

        indices = vector_index.search(vectors[[0], :], k=10)
        print(indices.flatten())
        print(ids)
        self.assertTrue(np.isin(indices.flatten(), ids).all().item())
        self.assertEqual((1, 10), indices.shape)

    def test_search_with_return_distance(self):
        vector_index = self.get_vector_index()
        vectors = self._get_random_data()
        vector_index.build(vectors)

        indices, distances = vector_index.search(vectors[[0], :], k=10, return_distance=True)
        self.assertEqual((1, 10), indices.shape)
        self.assertEqual((1, 10), distances.shape)

    def test_search_with_k_larger_than_index(self):
        vector_index = self.get_vector_index()
        vectors = self._get_random_data()
        vector_index.build(vectors)

        with self.assertRaisesRegex(ValueError, 'Searching the vector index failed.'):
            vector_index.search(vectors[[0], :], k=vectors.shape[0]+1)
