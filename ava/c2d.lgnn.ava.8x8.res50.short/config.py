import os
import os.path as osp

import pyaction
from pyaction.configs.base_config import BaseConfig

_config_dict = dict(
    BN=dict(
        USE_PRECISE_STATS=False, 
        NUM_BATCHES_PRECISE=200, 
        MOMENTUM=0.1, 
        WEIGHT_DECAY=0.0,
    ),
    TRAIN=dict(
        DATASET="ava", 
        BATCH_SIZE=64, 
        EVAL_PERIOD=5, 
        CHECKPOINT_PERIOD=1,
        CHECKPOINT_FILE_PATH=osp.join(
            "/",
            *osp.realpath(pyaction.__file__).split("/")[:-2],
            "model_zoo/ava/pretrain/C2D_8x8_R50.pkl",
        ),
        CHECKPOINT_TYPE='caffe2'
    ),
    DATA=dict(
        PATH_TO_DATA_DIR=osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/kinetics"
        ),
        NUM_FRAMES=4,
        SAMPLING_RATE=16,
        INPUT_CHANNEL_NUM=[3],
    ),
    AVA=dict(
        ANNOTATION_DIR = (
            osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/ava/annotations/")
        ),
        BGR = False,
        DETECTION_SCORE_THRESH=0.9,
        EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv",
        FRAME_DIR=osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/ava/frames"
        ),
        FRAME_LIST_DIR = (
            osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/ava/frame_lists")
        ),
        FULL_TEST_ON_VAL = False,
        GROUNDTRUTH_FILE = "ava_val_v2.2.csv",
        IMG_PROC_BACKEND = "cv2",
        LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt",
        TEST_FORCE_FLIP = False,
        TEST_LISTS = ["val.csv"],
        TEST_PREDICT_BOX_LISTS=[
            "person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv"
        ],
        TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"],
        TRAIN_LISTS = ["train.csv"],
        TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229],
        TRAIN_PCA_EIGVEC = [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ],
        TRAIN_PCA_JITTER_ONLY = True,
        TRAIN_PREDICT_BOX_LISTS=[
            "ava_train_v2.2.csv",
            "person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv"
        ],
        TRAIN_USE_COLOR_AUGMENTATION = False,
    ),
    DETECTION=dict(
        ENABLE = True,
        # Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
        ALIGNED = True,
        # Spatial scale factor.
        SPATIAL_SCALE_FACTOR = 16,
        # RoI tranformation resolution.
        ROI_XFORM_RESOLUTION = 7,
    ),
    RESNET=dict(
        ZERO_INIT_FINAL_BN=True, 
        DEPTH=50, 
        NUM_BLOCK_TEMP_KERNEL=[[3], [4], [6], [3]],
    ),
    # NONLOCAL=dict(
    #     LOCATION=[[[]], [[1, 3]], [[1, 3, 5]], [[]]],
    #     GROUP=[[1], [1], [1], [1]],
    #     INSTANTIATION="softmax",
    # ),
    LATENTGNN3D=dict(
        LOCATION=[
            [[]], 
            [[1, 3]], 
            [[1, 3, 5]], 
            [[]]],
        GROUP=[[1], [1], [1], [1]],
        NUM_NODES=[
            # Res2
            [0],
            # Res3
            [200],
            # Res4
            [200],
            # Res5
            [0],
        ],
        INSTANTIATION="softmax",
        CONFIG=dict(
            channel_stride=2,
            mode='asymmetric',
            # norm_func = F.normalize,
            latent_nonlocal = True,
            norm_type = 'batchnorm',
            latent_value_transform = True,
            latent_skip = True,
            zero_init_final_norm=True,
            norm_eps=1e-5,
            norm_momentum=0.1,
        )
    ),
    SOLVER=dict(
        BASE_LR=0.1, 
        LR_POLICY="steps_with_relative_lrs",
        STEPS = [0, 10, 15, 20],
        LRS = [1, 0.1, 0.01, 0.001],
        MAX_EPOCH=20, 
        MOMENTUM=0.9,
        WEIGHT_DECAY=1e-7,
        WARMUP_EPOCHS=5,
        WARMUP_START_LR=0.000125,
        OPTIMIZING_METHOD='sgd'),
    MODEL=dict(
        ARCH="slowonly",
        LOSS_FUNC='bce', 
        NUM_CLASSES=80,),
    TEST=dict(
        ENABLE=True, 
        DATASET="ava", 
        BATCH_SIZE=4),
    DATA_LOADER=dict(
        NUM_WORKERS=16,
        PIN_MEMORY=True,
        # ENABLE_MULTI_THREAD_DECODE=True
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
