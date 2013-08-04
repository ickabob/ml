import itertools

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 2, 'x') --> ABC DEF Gxx
    return itertools.izip_longest(*(iter(iterable),)*n, fillvalue=fillvalue)

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))
