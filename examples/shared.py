from sklearn.metrics import f1_score


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    print('Train accuracy: {:.2f}'.format(
        f1_score(y_pred, train.y, average='micro')))
    print('Test accuracy: {:.2f}'.format(f1_score(y_pred_test, test.y, average='micro')))
    print('---')
