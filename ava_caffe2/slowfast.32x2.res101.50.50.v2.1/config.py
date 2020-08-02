import os
import os.path as osp

import pyaction
from pyaction.configs.base_config import BaseConfig

_config_dict = dict(
    AVA=dict(
        ANNOTATION_DIR = (
            osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/ava/annotations/v2.1")
        ),
        BGR = False,
        DETECTION_SCORE_THRESH=0.8,
        EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.1.csv",
        FRAME_DIR=osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/ava/frames"
        ),
        FRAME_LIST_DIR = (
            osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/ava/frame_lists")
        ),
        FULL_TEST_ON_VAL = False,
        GROUNDTRUTH_FILE = "ava_val_v2.1.csv",
        IMG_PROC_BACKEND = "cv2",
        LABEL_MAP_FILE = "ava_action_list_v2.1_for_activitynet_2018.pbtxt",
        TEST_FORCE_FLIP = False,
        TEST_LISTS = ["val.csv"],
        TEST_PREDICT_BOX_LISTS=[
            "../person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv"
        ],
        # TEST_LISTS = ["val_subset_5.csv"],
        TRAIN_GT_BOX_LISTS = ["ava_train_v2.1.csv"],
        TRAIN_LISTS = ["train.csv"],
        TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229],
        TRAIN_PCA_EIGVEC = [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ],
        TRAIN_PCA_JITTER_ONLY = True,
        TRAIN_PREDICT_BOX_LISTS=[
            # "ava_train_v2.2.csv",
            # "person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv"
        ],
        TRAIN_USE_COLOR_AUGMENTATION = False,
    ),
    BN=dict(
        USE_PRECISE_STATS=False, 
        NUM_BATCHES_PRECISE=200, 
        MOMENTUM=0.1, 
        WEIGHT_DECAY=0.0,
    ),
    DATA=dict(
        PATH_TO_DATA_DIR='',
        NUM_FRAMES=32,
        SAMPLING_RATE=2,
        INPUT_CHANNEL_NUM=[3, 3],
    ),
    DATA_LOADER=dict(
        NUM_WORKERS=2,
        PIN_MEMORY=True,
        # ENABLE_MULTI_THREAD_DECODE=True
    ),
    DETECTION=dict(
        ENABLE = True,
        # Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
        ALIGNED = False,
        # Spatial scale factor.
        SPATIAL_SCALE_FACTOR = 16,
        # RoI tranformation resolution.
        ROI_XFORM_RESOLUTION = 7,
    ),
    MODEL=dict(
        ARCH="slowfast",
        LOSS_FUNC='bce', 
        NUM_CLASSES=80,),

    NONLOCAL=dict(
        LOCATION=[[[], []], [[], []], [[6, 13, 20], []], [[], []]],
        GROUP=[[1, 1], [1, 1], [1, 1], [1, 1]],
        INSTANTIATION='dot_product',
        POOL=[[[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]]]
    ),
    RESNET=dict(
        DEPTH=101, 
        NUM_BLOCK_TEMP_KERNEL= [[3, 3], [4, 4], [6, 6], [3, 3]],
        SPATIAL_DILATIONS = [[1, 1], [1, 1], [1, 1], [2, 2]],
        SPATIAL_STRIDES = [[1, 1], [2, 2], [2, 2], [1, 1]],
        ZERO_INIT_FINAL_BN=True, 
    ),
    SLOWFAST=dict(
        ALPHA=4,
        BETA_INV=8,
        FUSION_CONV_CHANNEL_RATIO=2,
        FUSION_KERNEL_SZ=5,
    ),
    SOLVER=dict(
        BASE_LR=0.1, 
        LR_POLICY="cosine",
        STEPS = [],
        LRS = [],
        # LRS = [1, 0.1, 0.01, 0.001],
        MAX_EPOCH=300, 
        MOMENTUM=0.9,
        WEIGHT_DECAY=1e-7,
        WARMUP_EPOCHS=0,
        WARMUP_START_LR=0.01,
        OPTIMIZING_METHOD='sgd'),
    
    TEST=dict(
        ENABLE=True, 
        DATASET="ava", 
        BATCH_SIZE=4,
        CHECKPOINT_TYPE="caffe2",
        CHECKPOINT_FILE_PATH=osp.join(
            "/",
            *osp.realpath(pyaction.__file__).split("/")[:-2],
            "model_zoo/ava/SLOWFAST_32x2_R101_50_50_v2.1.pkl",
        ),
    ),
    TRAIN=dict(
        ENABLE=False,
        DATASET="ava", 
        BATCH_SIZE=16, 
        EVAL_PERIOD=1, 
        CHECKPOINT_PERIOD=1,
        CHECKPOINT_FILE_PATH='',
        CHECKPOINT_TYPE='caffe2'
    ),
    NUM_GPUS=4,
    NUM_SHARDS=1,
    RNG_SEED=0,
    OUTPUT_DIR=osp.join(
        os.getenv("PYACTION_OUTPUT"),
        "model_logs",
        *osp.realpath(__file__).split("/")[-3:-1],
    ),
)


class KineticsConfig(BaseConfig):
    def __init__(self):
        super(KineticsConfig, self).__init__()
        self._register_configuration(_config_dict)


config = KineticsConfig()
