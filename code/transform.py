import random

import numpy as np
import torch


class Compose(object):
    """Composes several transforms together.

    Inspired by `torchvision.transform.Compose`

    Parameters
    ----------
    transforms : list
        list of transforms to compose.
    unpack : bool
        Force unpacking of received argument list/tuple. It is useful when you
        use `torchnet.dataset.TranformDataset`.

    Example
    -------
        >>> Compose([
        >>>     SubsampleSequence(10),
        >>>     ToTensor(),
        >>> ])

    """
    def __init__(self, transforms, unpack=False):
        self.transforms = transforms
        self.unpack = unpack

    def __call__(self, *args):
        if (self.unpack and len(args) == 1 and
           (isinstance(args[0], tuple) or isinstance(args[0], list))):
            args = tuple(*args)

        for t in self.transforms:
            args = t(*args)
        return args


class ProposalMask(object):
    """Zero-out contributions of proposals at the beggining

    We generate proposals at the beginning of the sequence without meaningfull
    interpretation. We generate a mask to zero-out its contribution during
    training.
    Expected label must be a ndarray. Thus, mask is also a ndarray with similar
    shape and its appends to the current argument list (*args)

    Parameters
    ----------
    label_idx : int
        Index of label tensor inside *args

    """
    def __init__(self, label_idx=1, num_proposals=16):
        self.label_idx = label_idx
        self.num_proposals = num_proposals

    def __call__(self, *args):
        y = args[self.label_idx]
        mask = np.ones_like(y)
        for j in range(self.num_proposals - 1):
            if j >= y.shape[0]:
                break
            mask[j, j+1:] = 0
        return tuple([i for i in args] + [mask])


class SubsampleSequence(object):
    """Return a subset of length max-length of a sequence

    The subset is taking out sequentially from the first dimension of the
    sequence until reach max-length.
    Additional arguments must support slicing and provide its length.

    Arguments
    ---------
    max_length : int
        Max length of the sequence
    if_short_fn : function, optional
        Workaround to fulfill max-length requirement. Function should receive
        a single argument and max-length.

    Raises
    ------
    ValueError
        sequence length less than max_length

    """
    def __init__(self, max_length=512, if_short_fn=None):
        self.max_length = max_length
        self.if_short_fn = if_short_fn

    def __call__(self, *args):
        seq_length = len(args[0])
        if seq_length < self.max_length and self.if_short_fn is None:
            raise ValueError('Sequence length ({}) shorter than {}.'.format(
                seq_length, self.max_length))
        if seq_length < self.max_length:
            args = [self.if_short_fn(i, self.max_length) for i in args]

        start_point = random.randint(0, max(0, seq_length - self.max_length))
        subset = slice(start_point,
                       min(start_point + self.max_length, seq_length))
        return tuple([i[subset, ...] for i in args])


class ToTensor(object):
    """Equivalent to `torchvision.transform.ToTensor` over multiple arguments
    """
    def __call__(self, *args):
        argout = []
        for i in args:
            if isinstance(i, np.ndarray):
                argout.append(torch.from_numpy(i))
            elif torch.is_tensor(i):
                argout.append(i)
            else:
                raise ValueError('Unsupported type {}'.format(type(i)))
        return tuple(argout)


def tile_sequence(arr, min_length):
    """Tile sequence along first dimension

    Arguments:
        arr (ndarray) : t x d. Time information is in the first axis.
        min_length (int) : mininum length for the sequence

    Outputs:
        arr_tiled (ndarray) : t x d. Tiled array with t >= min_length.

    """
    if isinstance(arr, np.ndarray):
        num_reps = int(np.ceil(1.0 * min_length / arr.shape[0]))
        arr_tiled = np.tile(arr, (num_reps,) + (1,) * len(arr.shape[1:]))
        return arr_tiled[:min_length, ...]
