from __future__ import with_statement
from __future__ import absolute_import
from __future__ import division
import json

import numpy as np
import torch
from torchnet.dataset import BatchDataset
from torchnet.dataset.dataset import Dataset
from io import open


class IndexedDataset(Dataset):
    """Dataset load data from general indexed data structures e.g. dict.
    """
    def __init__(self, map_index, *args):
        """Setup IndexedDataset

        Parameters
        ----------
        map_index : dict or str
            Map integer id to valid index for the (multiple) data structures
            with your data.

        """
        self.data = [i for i in args]
        if not isinstance(map_index, dict):
            with open(map_index, u'r') as f:
                map_index = json.load(f)
        self.dict_of_indexes = map_index

    def __getitem__(self, index):
        idx = self.dict_of_indexes[index]
        return tuple(i[idx] for i in self.data)

    def __len__(self):
        return len(self.dict_of_indexes)

class UntrimmedSequenceDataset(Dataset):
    """Dataset for dealing with variable length sequences and their labels

    This dataset wrap variable length sequential observations by trimming them
    and their labels on demand. Sequential observations are expected to be
    located along the first axis. Randomization is strongly recommended if the
    labels from each sequence are uniform. Depending on your batch-size and
    sequence lenghts, a sequential sampler may load many instances of the same
    sequence in the same batch.
    Expected arguments must be dictionaries with ndarray as values.
    Outputs from ``__getitem__`` are ndarray.

    Arguments:
        max_length (int) : legnth of sequences returned by this dataset.
        stride (int) : stride used to grab fixed-length sequences out of longer
            longer sequence.
        if_short_fn (function) : Workaround to fulfill max-length
            requirement for short sequences. Function must receive
            a single argument and min-length := `max_length`. Pass None to
            raise error when sequence is shorter than `max_length`.

    Raises:
        ValueError :
            - sequence length less than max_length when `if_short_fn` is
              `None`.
    """
    def __init__(self, max_length, stride, if_short_fn, *args):
        self.max_length = max_length
        self.if_short_fn = if_short_fn
        if stride > 0:
            self.stride = stride
        else:
            raise ValueError('Stride must be greater than zero')

        # Record sequence ids and length of each sequence
        if isinstance(args[0], dict):
            self.seq_ids, seq_lengths = list(
                zip(*[(key, np.array(value).shape[0])
                      for key, value in args[0].items()]))
        else:
            raise ValueError('Unsupported data type: {}'.format(type(args[0])))

        # Each bucket corresponds to the maximum number of elements that we can
        # retrieve from each sequence
        # max(1, l - max_length + 1) only place windows with max_length guard
        self.buckets = np.cumsum(
            [int((max(1, l - max_length + 1) - 1)/stride + 1)
             for l in seq_lengths])

        # Placing data (shoud we make a method for this?)
        self.data = tuple([i for i in args])

        # Catch bugs and avoid criptic errors
        ref_keys = set(self.data[0])
        print len(ref_keys)
        print [len(set(i)) for i in self.data]
        msg = "Dictionaries must have the same set of keys"
        assert all([set(i) == ref_keys for i in self.data[1:]]), msg

    def __getitem__(self, index):
        if index < 0:
            raise IndexError('Negative index are not supported.')
        elif index >= len(self):
            raise IndexError('Index out of bounds')

        # Binary search: get index of seq-id associated with query index
        seq_index = np.searchsorted(self.buckets, index, 'right')

        # Compute idx inside butcket-id
        l_before = self.buckets[seq_index - 1]
        if seq_index == 0:
            l_before = 0
        idx_i = (index - l_before) * self.stride

        idx_f = min(idx_i + self.max_length,
                    self.data[0][self.seq_ids[seq_index]].shape[0])

        argout = []
        for value in self.data:
            # Retrieve data from that bucket from the idx ongoing
            data = value[self.seq_ids[seq_index]][idx_i:idx_f, ...]
            seq_length = data.shape[0]

            # Apply if_short_fn for data shorter than max-length
            if seq_length < self.max_length and self.if_short_fn is None:
                raise ValueError(
                    'Sequence length ({}) shorter than {}.'.format(
                        seq_length, self.max_length))
            if seq_length < self.max_length:
                data = self.if_short_fn(data, self.max_length)

            argout.append(data)
        return tuple(argout)

    def __len__(self):
        return self.buckets[-1]


class VariedLengthBatchDataset(BatchDataset):
    """BatchDataset for varied length sequences

    Notes
    -----
        - Hack to do call our own 'makebatch' function
        - It ignores merge function

    """
    def __init__(self, *args, **kwargs):
        """Setup VariedLengthBatchDataset as BatchDataset


        Parameters
        ----------
        dataset : Dataset
            Dataset to be batched.
        batchsize : int
            Size of the batch.
        perm : function, optional
            Function used to shuffle the dataset before batching.
            `perm(idx, size)` should return the shuffled index of
            `idx`th sample. By default, the function is the identity.
        policy : str, optional
            Policy to handle the corner cases when the underlying dataset size
            is not divisible by `batchsize`. One of (`include-last`,
            `skip-last`, `divisible-only`).
            - `include-last` makes sure all samples of the underlying dataset
               will be seen, batches will be of size equal or inferior to
               `batchsize`.
            - `skip-last` will skip last examples of the underlying dataset if
               its size is not properly divisible. Batches will be always of
               size equal to `batchsize`.
            - `divisible-only` will raise an error if the underlying dataset
               has not a size divisible by `batchsize`.
       filter : function, optional
            Function to filter the sample before batching. If `filter(sample)`
            is True, then sample is included for batching. Otherwise, it is
            excluded. By default, `filter(sample)` returns True for any
            `sample`.

        """
        super(VariedLengthBatchDataset, self).__init__(*args, **kwargs)
        self.makebatch = makebatch_varied_seqlen

    def __getitem__(self, idx):
        # Code duplication because Base class will return. Suggestions?
        if idx >= len(self):
            raise IndexError(u"CustomRange index out of range")
        maxidx = len(self.dataset)

        samples, max_length = [], 0
        for i in xrange(0, self.batchsize):
            j = idx * self.batchsize + i
            if j >= maxidx:
                break

            j = self.perm(j, maxidx)
            sample = self.dataset[j]

            if torch.is_tensor(sample[0]):
                seq_length = sample[0].size()[0]
            else:
                raise ValueError(u'Invalid data type: {}'.format(type(sample)))

            if seq_length > max_length:
                max_length = seq_length

            if self.filter(sample):
                samples.append(sample)

        samples = self.makebatch(samples, max_length)
        return samples


def makebatch_varied_seqlen(samples, max_length):
    """Stack a list of varied length samples

    Parameters
    ----------
    samples : list

    Returns
    -------
    stacked_samples : list

    """
    def zero_padding(data, max_length=max_length):
        size_d = data.size()
        pad = torch.zeros(max(0, max_length - size_d[0],),
                          *size_d[1:]).type_as(data)
        return pad

    if not isinstance(samples[0], tuple):
        NotImplementedError(u'Unknown input sample: {}'.format(
            type(samples[0])))

    batched_data = []
    for j in xrange(len(samples[0])):
        batched_data.append(
            torch.stack([
                torch.cat((i[j], zero_padding(i[j])))
                for i in samples]))
    return batched_data
