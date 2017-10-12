from argparse import ArgumentParser

RECORD_HYP_PRM = [
    'learning_rate', 'lr_decay', 'num_decay', 'w_background', 'w0', 'w1',
    'lambda_joint', 'lambda_loc', 'lambda_prop', 'lambda_segcls', 'dropout',
    'weight_decay', 'sequence_stride']


def data_eval_arguments(p):
    """Dataset evaluation/demo arguments
    """
    p.add_argument('-ds', '--dataset-file', required=True,
                   help=('Filename with of pre-forwarded vis feats (hickle format)'))
    p.add_argument('-dsr', '--data-sampling-rate', default=16, type=int,
                   help=('Sampling rate (as number of frames) used to get '
                         'observations such as vis features'))
    # Evaluation criteria
    p.add_argument('-tiou', '--tiou-threshold', nargs='+', default=[0.5],
                   help='tIoU threshold values to define matching')

    # Transformations / Helpers
    p.add_argument('-vif', '--video-info-file',
                   default='annotations/metadata/val_list.tsv',
                   help=('TSV file with info about videos. It transforms '
                         'frame predictions to seconds'))
    p.add_argument('-ff', '--filter-file', default='filter-file.tsv',
                   help=('TSV file with list of videos to evaluate. It '
                         'evaluates the model only on videos in this file.'))
    p.add_argument('-dbs', '--dataset', default='thumos14',
                   choices=['thumos14', 'other'],
                   help='Dataset used for evaluation')
    p.add_argument('-gtf', '--groundtruth-file',
                   help=('GT filename with annotations of the dataset. '
                         'Dirname of annotations for THUMOS14.'),
                   default='annotations/thumos_gt.json')
    p.add_argument('-dbp', '--database-subset', default='validation',
                   help=('Indicate evaluation subset if any. (val, test)` for'
                         'THUMOS14.'))
    p.add_argument('-dbif', '--database-class-index-file',
                   default='annotations/metadata/class_index_detection.tsv',
                   help='TSV file with class index mapping')
    p.add_argument('-dbot', '--database-detection-template',
                   default='data/template_thumos14_detection.json',
                   help='Template for fill in detections')


def misc_arguments(p):
    p.add_argument('-v', '--verbose', default=0, type=int,
                   help='Verbosity level')
    p.add_argument('-gpu', '--gpu-device', default=0, type=int,
                   help='GPU device ID')


def model_arguments_short(p):
    p.add_argument('-ww', '--window-width', default=256, type=int,
                   help=('training window width (should be greater than '
                         'num_proposals)'))
    p.add_argument('-dp', '--dropout', default=0, type=float,
                   help='Dropout probabilty. If 0, no dropout layer is used.')
    p.add_argument('-np', '--num-proposals', default=256, type=int,
                   help='Number of proposals')
    p.add_argument('-nl', '--num-layers', default=2, type=int,
                   help='no. LSTMs to use')
    p.add_argument('-hs', '--hidden-state-size', default=256, type=int,
                   help='width of LSTM layer')


def postprocessing_arguments(p):
    p.add_argument('-thr', '--threshold', default=0.05, type=float,
                   help='Keep detections with confidence >= this value')
    p.add_argument('-nms', '--nms-threshold', default=0.7, type=float,
                   help='NMS threshold over detections')
    p.add_argument('-mdv', '--max-per-video', default=100, type=int,
                   help='Maximum number of detections per video')


def tensorboard_arguments(p):
    p.add_argument('-tbd', '--tensorboard-dir', default='tblog',
                   help='Tensorboard root')
    p.add_argument('-tbhp', '--tensorboard-log-hypprm', nargs='+',
                   default=RECORD_HYP_PRM,
                   help='list of hyper-parameters to log')
    p.add_argument('-tbft', '--tb-flush-time', type=int, default=60,
                   help='Flush time for tensorboard update')

