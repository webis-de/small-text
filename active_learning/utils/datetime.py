def format_timedelta(td):
    if td.days < 0:
        raise ValueError('timedelta must be positive')
    hours, sec = divmod(td.seconds, 3600)
    mins, sec = divmod(sec, 60)

    return f'{hours:02}:{mins:02}:{sec:02}'
