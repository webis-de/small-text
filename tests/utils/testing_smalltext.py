from small_text.data.datasets import DatasetView


class AnyDatasetView(object):

    def __eq__(self, other):
        return isinstance(other, DatasetView)
