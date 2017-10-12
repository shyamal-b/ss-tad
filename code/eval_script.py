from __future__ import with_statement

import argparse
import json
import os
import time
import sys
from six.moves import range

import hickle as hkl
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

import config
from segment import non_maxima_supression
from model import *



def get_segments(y, stride=16):
    """Return segments and classification score

    Arguments
        y (ndarray) : array of (L, K), with L = length of video discretized by
            `stride` (which is 16 for our model).
        stride (int) : temporal stride of vis encoder from video (e.g. "delta").

    Outputs
        dets (ndarray) : array of detections (L x K, 2) denoting  f-init and
        f-end
        scores (ndarray) : array of confidence score (L x K, C).

    """
    dets, scores = [], []
    L, K, C = y.shape
    for i, l in enumerate(range(0, L * stride, stride)):
        prop_l = (l + stride - 1) * np.ones((K, 2), dtype=np.float32)
        prop_l[:, 0] = l - np.arange(K) * stride
        curr_y = y[i, :, :]
        
        # Remove detections covering negative frames
        idx_valid = np.where(prop_l[:, 0] >= 0)[0]
        dets.append(prop_l[idx_valid, :])
        scores.append(curr_y[idx_valid, :])

    all_dets, all_scores = np.vstack(dets), np.vstack(scores)
    return all_dets, all_scores


def inference(X, model, gpu_device, idx_pcp=3):
    """Returns output of the model for X as ndarray

    Arguments
        idx_pcp (int, optional) : temporal arg to take index output returned by
            model. It must return Temporal-Class-Probability (L x K x C).
    """
    assert gpu_device >= 0, "must have gpu device for current code."
    with torch.cuda.device(gpu_device):
        torch.backends.cudnn.enabled = True
        X = X.astype(np.float32, copy=False)
        Xt = torch.from_numpy(np.expand_dims(X, 0))
        if gpu_device >= 0:
            Xt = Xt.pin_memory().cuda(gpu_device)
    
        data = Variable(Xt, volatile=True)
        output = model(data)
    
        if isinstance(output, tuple):
            predictions = output[idx_pcp].data[0]
        else:
            # Assuming output is Variable
            predictions = output.data[0]
    
        if predictions.is_cuda:
            predictions = predictions.cpu()
    
        pred_size = predictions.size()
        if len(pred_size) > 3 and pred_size[0] == 1:
            predictions.squeeze_(0)
    return predictions.numpy()

def load_features(filename):
    return hkl.load(filename)

def load_labels_index_file(filename):
    df = pd.read_table(filename)
    d_by_cols = df.to_dict('list')
    d_by_idx_label = dict(zip(d_by_cols['idx-label'],
                              d_by_cols['activity-label']))
    return d_by_idx_label

def dump_results(dataset, results, filename, class_index_dict, template):
    # You can customize this to dump in the format you need.
    raise ValueError("Need to add dataset dump formatter for dataset: '{}'".format(dataset))

def load_model(filename, gpu_id, total_gpus=8):
    print "loading model..."
#    mod = torch.load(filename, map_location={'cuda:{}'.format(i):'cuda:{}'.format(gpu_id) for i in xrange(total_gpus)})
    mod = torch.load(filename, map_location=lambda storage, loc: storage)
    print "success!"
    return mod


def load_video_info(filename):
    if os.path.isfile(filename):
        df = pd.read_table(filename)
        df.drop_duplicates('video-name', inplace=True)
        d_ = df.to_dict(orient='list')
        metadata_names = d_.keys()
        metadata_names.remove('video-name')
        d_gv = {video_id: {k: d_[k][i] for k in metadata_names}
                for i, video_id in enumerate(d_['video-name'])}
        return d_gv


def load_video_list(filename):
    if os.path.isfile(filename):
        return pd.read_table(filename)['video-name'].unique()


def postprocessing(predictions, thresh=0, nms_threshold=1.0, max_per_video=0):
    """Return detections per video
    """
    detections, scores = predictions

    # skip i = score.shape[1] because it's background class
    num_classes = scores.shape[1] - 1
    all_segments = [[] for _ in range(num_classes)]

    # Post-process detections for each non-background class.
    for i in range(num_classes):
        idx = np.where(scores[:, i] > thresh)[0]
        cls_det, cls_score = non_maxima_supression(
            detections[idx, :], scores[idx, i], nms_threshold,
            inc_unit=1.0)  # Set unit to zero for time
        all_segments[i] = np.hstack([cls_det, cls_score[:, np.newaxis]])

    # Optional: limit to max_per_video detections *over all classes*
    if max_per_video > 0:
        video_scores = np.hstack([all_segments[i][:, -1]
                                  for i in range(num_classes)])
        if len(video_scores) > max_per_video:
            video_thresh = np.sort(video_scores)[-max_per_video]
            for i in range(num_classes):
                keep = np.where(all_segments[i][:, -1] >= video_thresh)[0]
                all_segments[i] = all_segments[i][keep, :]
    return all_segments

example_dataset_eval_fn = None  # placeholder, update for your dataset.
# from thumos14.eval_detection import THUMOS14detection

DATASET_EVAL_METHODS = {
    "example_dataset_name" : example_dataset_eval_fn,
    # Note: update with your dataset accordingly
}


def eval_detections(dataset, gt_filename, rst_filename, tiou_range, subset,
                    verbose):
    performance = []
    eval_det_fn = DATASET_EVAL_METHODS[dataset]
    eval_detection = eval_det_fn(gt_filename, rst_filename, subset=subset)
    for tiou in sorted(tiou_range):
        if verbose: print("SETTING TIOU TO: {}".format(tiou))
        eval_detection.tiou_thr = float(tiou)
        mAP = eval_detection.evaluate()
        performance.append((tiou, mAP))
    return performance


def update_results(pile, results, video_id, video_dict=None):
    # Map frame idx values into seconds
    if isinstance(video_dict, dict):
        duration = video_dict[video_id]['duration']
        num_frames = video_dict[video_id]['num-frames']
        for i, _ in enumerate(results):
            results[i][:, 0] *= duration / num_frames
            results[i][:, 1] *= duration / num_frames
    pile[video_id] = results


def main(args):
    # Load class index file
    class_index_dict = load_labels_index_file(args.database_class_index_file)

    # Load video info
    video_info = load_video_info(args.video_info_file)

    # Load features
    X = load_features(args.dataset_file)

    # Only evaluate videos in filter file
    video_list = load_video_list(args.filter_file)
    if video_list is None:
        video_list = list(X.keys())

    # Load model, get epoch
    # Parameters of model
    model = load_model(args.model_file, args.gpu_device)
    if args.gpu_device >= 0:
        model = model.cuda(args.gpu_device)
    model.eval()

    # Iterate over video in features
    results = {}
    total_vids = len(video_list)
    for curr_vid_idx, video_id in enumerate(video_list):
        if args.verbose and args.verbose != 3:
            # print the progress bar
            sys.stdout.write('\r')
            prog_bar = 100.0/(total_vids-1) * curr_vid_idx
            sys.stdout.write("[%-100s] %d%% // currently at video %d = %s." % ('='*int(prog_bar), prog_bar, curr_vid_idx, video_id))
            sys.stdout.flush();
        predictions_i = inference(X[video_id], model, args.gpu_device,
                                  args.model_idx_output)
        # Raw (clean) detections
        detections_i = get_segments(predictions_i,
                                    stride=args.data_sampling_rate)
        # Post-processing
        results_i = postprocessing(
            detections_i, thresh=args.threshold,
            nms_threshold=args.nms_threshold, max_per_video=args.max_per_video)

    # Dump results
    dump_results(
        args.dataset, results, args.output_file,
        class_index_dict, args.database_detection_template)

    # Evaluation
    performance = eval_detections(
        args.dataset, args.groundtruth_file, args.output_file,
        args.tiou_threshold, args.database_subset, args.verbose)
    return 0

def input_parser():
    description = "SS-TAD inference over videos"
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-mf', '--model-file', required=True,
                   help=('Filename of pth (pickle torch file) with model '
                         'parameters'))
    p.add_argument('-of', '--output-file', default='non-existent',
                   help='Filename of output')
    p.add_argument('-mio', '--model-idx-output', default=3, type=int,
                   help=('Index of output of model with confidence/scores '
                         'after forward pass'))
    config.data_eval_arguments(p)
    config.postprocessing_arguments(p)
    config.misc_arguments(p)
    return p

if __name__ == '__main__':
    p = input_parser()
    args = p.parse_args()
    main(args)
