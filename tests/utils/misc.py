def increase_dense_labels_safe(ds):
    """Increase the labels without leaving the range of the target_labels property.
    The only purpose of this operation is to alter the labels so we can check for a change later."""

    # modulo needs not be used when single index result is 0
    if ds.y.max() == 0:
        ds.y = ds.y + 1
    else:
        ds.y = (ds.y + 1) % (ds.y.max() + 1)
    return ds
