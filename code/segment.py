import numpy as np


def iou(target_segments, test_segments, inc_unit=1.0):
    """Compute intersection over union btw segments

    Parameters
    ----------
    target_segments : ndarray.
        2d-ndarray of size [m, 2] with format [t-init, t-end].
    test_segments : ndarray.
        2d-ndarray of size [n x 2] with format [t-init, t-end].
    inc_unit : float
        Extra unit to include once length is computed. Set it to zero if your
        'f-end' should not be included or your are working with float values.

    Outputs
    -------
    iou : ndarray
        2d-ndarray of size [m x n] with tIoU ratio.

    Raises
    ------
    ValueError
        target_segments or test_segments are not 2d-ndarray.

    Notes
    -----
    It assumes that target-segments are more scarce that test-segments

    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    iou = np.empty((m, n))
    for i in xrange(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + inc_unit).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + inc_unit) +
                 (target_segments[i, 1] - target_segments[i, 0] + inc_unit) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        iou[i, :] = intersection / union
    return iou


def non_maxima_supression(dets, score=None, overlap=0.7, measure='iou',
                          inc_unit=1.0, return_indexes=False):
    """Non-maximum suppression
    Greedily select high-scoring detections and skip detections that are
    significantly covered by a previously selected detection.
    This version is translated from Matlab code by Tomasz Malisiewicz.

    Parameters
    ----------
    dets : ndarray.
        2d-ndarray of size [num-segments, 2]. Each row is ['f-init', 'f-end'].
    score : ndarray.
        1d-ndarray of with detection scores. Size [num-segments, 2].
    overlap : float, optional.
        Minimum overlap ratio.
    measure : str, optional.
        Overlap measure used to perform NMS either IoU ('iou') or ratio of
        intersection ('overlap')
    inc_unit : float
        Extra unit to include once length is computed. Set it to zero if your
        'f-end' should not be included or your are working with float values.
    return_indexes : bool
        Indexes of selected candidates

    Outputs
    -------
    pruned_dets : ndarray.
        Remaining after suppression.
    pruned_score : ndarray.
        Remaining after suppression.
    idx : list
        Indexes of pruned_dets and pruned_score

    Raises
    ------
    ValueError
        - Mismatch between score 1d-array and dets 2d-array
        - Unknown measure for defining overlap
        - f-init > f-end

    """
    measure = measure.lower()
    num_segments = dets.shape[0]
    if score is None:
        score = dets[:, 1]
    if score.shape[0] != num_segments:
        raise ValueError('Mismatch between dets and score.')
    if num_segments == 0 or num_segments == 1:
        if return_indexes:
            return dets, score, [0][:num_segments]
        return dets, score
    if (dets[:, 0] > dets[:, 1]).any():
        raise ValueError('f-init > f-end yield to infinite loop.')

    # Grab coordinates
    t1, t2 = dets[:, 0], dets[:, 1]
    area = t2 - t1 + inc_unit
    idx = np.argsort(score)
    pick = []
    while len(idx) > 0:
        last = len(idx) - 1
        i = idx[last]
        pick.append(i)

        tt1 = np.maximum(t1[i], t1[idx])
        tt2 = np.minimum(t2[i], t2[idx])

        wh = np.maximum(0.0, tt2 - tt1 + inc_unit)
        if measure == 'overlap':
            o = wh / area[idx]
        elif measure == 'iou':
            o = wh / (area[i] + area[idx] - wh)
        else:
            raise ValueError('Unknown overlap measure for NMS')

        idx = np.delete(idx, np.where(o > overlap)[0])

    pruned_dets = dets[pick, :]
    pruned_score = score[pick]
    if return_indexes:
        return pruned_dets, pruned_score, pick
    return pruned_dets, pruned_score
