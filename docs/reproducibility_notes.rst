=====================
Reproducibility Notes
=====================

In this section we provide notes on reproducing components that were implemented from scientific publications.

Preface
=======

Small-Text is specifically intended to provide a robust set of resuable pre-implemented components
which support the reproduction of scientific experiments.

However, in the end the correctness of your experiments is a serious matter and must still be assured:

1. **Never assume the implementation is perfect** : We might miss a special case as well (or break something accidentally).
   In other cases underlying functions from other libraries might have changed / might have a bug.

2. **Never assume default parameters are what you want** : It might be possible that this is the case,
   in most cases we will try to achieve this, but is it your responsibility to verify this.
   (In cases where a paper describes multiple configurations, the perfect default parameters might just not be possible.)

Nevertheless, using a shared code base (and posing questions / providing feedback on github)
will reduce the risk of errors compared to re-implementing these strategies yourself,
especially when more and more people will have reviewed the code.

Query Strategies
================

Expected Gradient Length
------------------------

:py:class:`~small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLength`
technically works with all common neural networks. In the context of active learning for
text classification it has been shown to work in combination with the
:py:class:`~small_text.integrations.pytorch.models.kimcnn.KimCNN` model [ZLW17]_
and also with transformer models [EHG+20]_. For the latter it is, however, costly in terms of runtime.
