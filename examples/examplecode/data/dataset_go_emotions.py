def get_go_emotions_dataset():
    import datasets
    from datasets import concatenate_datasets
    go_emotions = datasets.load_dataset('go_emotions')

    return concatenate_datasets([go_emotions['train'], go_emotions['validation']]), go_emotions['test']
