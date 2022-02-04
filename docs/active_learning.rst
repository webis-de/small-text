===============
Active Learning
===============

Active learning aims at creating training data for classification algorithms in a very efficient manner,
for cases in which a large amount of unlabeled data is available but labels are not.
Labeling such data is usually time-consuming and expensive.
To avoid having to label the full dataset,
active learning selectively chooses data points that are assumed to improve the model.
This is done iteratively, in a process that alternates between an algorithm selecting data to label,
and a human annotator who assigns the true labels to given samples.
The goal here is to maximize the quality of the model while keeping the annotation efforts at a minimum.
A comprehensive introduction to active learning can be found in (Settles, 2010) [Set10]_.

.. note:: Active learning is also a technical term in education.
          To avoid any confusion: This project is concerned with active learning
          in the context of `machine learning <https://en.wikipedia.org/wiki/Machine_learning>`_.


Components
==========

An active learning process can encompasses several, usually interchangeable components:
An :doc:`initalization strategy<components/initialization>`,
a :doc:`query strategy<components/query_strategies>`,
and (optionally) a :doc:`stopping criterion<components/stopping_criteria>`.

----

**References**

.. [Set10] Burr Settles. 2010.
   Active Learning Literature Survey.
   Computer Sciences Technical Report 1648.
   University of Wisconsin–Madison.