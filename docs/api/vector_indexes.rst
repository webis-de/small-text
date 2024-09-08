===============
VectorIndex API
===============

Given a vector, a vector index allows to search for similar other vectors.
This API makes several existing vector indexes usable in a unified manner.

.. contents:: Overview
   :depth: 1
   :local:
   :backlinks: none

----

Base
====

.. currentmodule:: small_text.vector_indexes.base

.. autoclass:: VectorIndex
   :members: build, remove, search, index

.. autoclass:: VectorIndexFactory
   :members: new
   :member-order: bysource
   :special-members: __init__

Implementations
===============

.. currentmodule:: small_text.vector_indexes.knn

.. autoclass:: KNNIndex
   :members: build, remove, search, build, remove, search
   :member-order: bysource
   :special-members: __init__
   :noindex:

.. currentmodule:: small_text.vector_indexes.hnsw

.. autoclass:: HNSWIndex
   :members: build, remove, search, build, remove, search
   :member-order: bysource
   :special-members: __init__
   :noindex:
