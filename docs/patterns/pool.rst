===============
Data Management
===============

.. note:: The following considerations are more relevant to Active Learning applications
          than to the experimental scenario.

In order to decouple the :py:class:`~active_learning.active_learner.PoolBasedActiveLearner`
from your data source, most of its methods operate solely
using indices. This also means, if data within your data source changes, e.g.,
examples are added or removed, or labels have been changed, you will need to update this information
within your :py:class:`~active_learning.active_learner.PoolBasedActiveLearner`.

Adding / Removing Data
======================

Whenever the dataset changes you need to (re-)initialize the active learner:

:py:meth:`~active_learning.active_learner.PoolBasedActiveLearner.initialize_data`

By default, this als trigger a re-training of the model, which can be suppressed
by passing :code:`retrain=True`.

Updating Labels
===============

:py:meth:`~active_learning.active_learner.PoolBasedActiveLearner.update_label_at`
:py:meth:`~active_learning.active_learner.PoolBasedActiveLearner.remove_label_at`
