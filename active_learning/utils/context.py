class NullProgressBar(object):

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, n):
        pass


def build_pbar_context(pbar_type, tqdm_kwargs=dict()):
    if pbar_type == 'tqdm':
        from tqdm import tqdm
        pbar_context = tqdm(**tqdm_kwargs)
    else:
        pbar_context = NullProgressBar()

    return pbar_context
