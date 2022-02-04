from sklearn.metrics import f1_score


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    print('Train accuracy: {:.2f}'.format(
        f1_score(y_pred, train.y, average='micro')))
    print('Test accuracy: {:.2f}'.format(f1_score(y_pred_test, test.y, average='micro')))
    print('---')


def evaluate_multi_label(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    # https://github.com/scikit-learn/scikit-learn/issues/18611
    print('Train accuracy: {:.2f}'.format(
        f1_score(y_pred.toarray(), train.y.toarray(), average='micro')))
    print('Test accuracy: {:.2f}'.format(f1_score(y_pred_test.toarray(),
                                                  test.y.toarray(), average='micro')))
    print('---')
