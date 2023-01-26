===============
Data Management
===============

Whenever the labeled data changes your current model might need retraining to reflect the updated data.
To decouple the :py:class:`~small_text.active_learner.PoolBasedActiveLearner`
from the application logic, most of its methods operate on indexes relative to the dataset, rather than on the dataset itself.
This also means if your data changes, i.e.,
examples are added, replaced or removed, or labels have been changed, you will need to keep this information updated.

.. note:: The following methods are more relevant to Active Learning applications
          than to the experimental scenario.

Updating Labels
===============

In case you want to revise or undo a past labeling, previous labels can be updated (:py:meth:`~small_text.active_learner.PoolBasedActiveLearner.update_label_at`)
or removed (:py:meth:`~small_text.active_learner.PoolBasedActiveLearner.remove_label_at`).

Adding / Removing Data
======================

Whenever the dataset changes, previous indices might be invalid and as a consequence the active learner need to be (re-)initialized:

:py:meth:`~small_text.active_learner.PoolBasedActiveLearner.initialize_data`

By default, this also triggers a re-training of the model, which can be suppressed
by passing :code:`retrain=True`.

Ignoring Data
=============

In real-world applications, there is often noisy data (e.g., after an OCR step). For this and other scenarios
in which you don't want to assign any labels (and also don't want to see this sample against in the next query),
you can ignore samples:

:py:meth:`~small_text.active_learner.PoolBasedActiveLearner.ignore_sample_at`
