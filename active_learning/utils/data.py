def list_length(x):
    if hasattr(x, 'shape'):
        return x.shape[0]
    else:
        return len(x)
