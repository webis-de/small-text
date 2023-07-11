from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from small_text import PytorchTextClassificationDataset

from examplecode.data.corpus_twenty_news import get_twenty_newsgroups_corpus


def get_train_test():
    return get_twenty_newsgroups_corpus(categories=['rec.sport.baseball', 'sci.med', 'rec.autos'])


def preprocess_data(train, test, pretrained_vectors, vocab_size=200_000, unk_token='[UNK]', pad_token='[PAD]'):
    num_special_tokens = 2
    vocab = {
        pretrained_vectors.index2word[i - num_special_tokens]: i
        for i in range(2, vocab_size)
    }

    vocab[unk_token] = 0
    vocab[pad_token] = 1

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])

    #trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=[unk_token, pad_token])
    #tokenizer.train_from_iterator(train.data, trainer=trainer)

    ds_train = PytorchTextClassificationDataset.from_arrays(train.data, train.target, tokenizer)
    ds_test = PytorchTextClassificationDataset.from_arrays(test.data, test.target, tokenizer)

    return ds_train, ds_test, tokenizer
