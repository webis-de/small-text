class Metric(object):
    """Represents an arbitrary metric.
    """

    def __init__(self, name, dtype=float, lower_is_better=True):
        """
        Parameters
        ----------
        name : str
            A name for the metric.
        dtype : any type, default=float
            Data type of the metric.
        lower_is_better : bool
            Indicates if the metric is better for lower values if True,
            otherwise it is assumed that higher values are better.
        """
        self.name = name
        self.dtype = dtype
        self.lower_is_better = lower_is_better

    def __repr__(self):
        return f'Metric(\'{self.name}\', dtype={self.dtype.__name__}, ' \
               f'lower_is_better={self.lower_is_better})'
