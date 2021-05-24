from sklearn.datasets import fetch_20newsgroups


def get_twenty_newsgroups_corpus(categories=['rec.sport.baseball', 'rec.sport.hockey']):

    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                               categories=categories)

    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                              categories=categories)

    return train, test
