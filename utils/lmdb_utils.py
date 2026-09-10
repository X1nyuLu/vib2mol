"""Helpers for LMDB datasets with numeric record keys and metadata."""

import pickle


LENGTH_KEY = b'length'


def read_lmdb_length(txn):
    """Read the cached record count, falling back for legacy databases."""
    value = txn.get(LENGTH_KEY)
    if value is not None:
        length = pickle.loads(value)
        if type(length) is not int or length < 0:
            raise ValueError(f'Invalid LMDB length metadata: {length!r}')
        return length
    return txn.stat()['entries']


def iter_lmdb_records(txn):
    """Yield contiguous numeric keys in the same order as lmdbDataset.

    Cursor order is lexicographic (0, 1, 10, ...), not dataset index order.
    Mixing these orders silently associates retrieval scores with wrong records.
    """
    for index in range(read_lmdb_length(txn)):
        key = str(index).encode('ascii')
        value = txn.get(key)
        if value is None:
            raise ValueError(f'Missing LMDB record {key!r}; expected contiguous numeric keys')
        yield key, value
