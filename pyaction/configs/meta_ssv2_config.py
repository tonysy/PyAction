import os
import os.path as osp

import pyaction
from pyaction.configs.base_config import BaseConfig

pyaction_home = os.path.dirname(pyaction.__path__[0])
# curr_folder = osp.realpath(__file__)[:-9]

_config_dict = dict(
    META=dict(
        ENABLE=True,
        SETTINGS=dict(
            # Setting
            N_SUPPORT_WAY=5,
            K_SUPPORT_SHOT=1,
            N_QUERY_WAY=1,
            K_QUERY_SHOT=1,
        ),
        DATA=dict(
            # either '_repalced' or ''
            CSV_SUFFIX="",
            # Task related
            NUM_TRAIN_TASKS=1500,
            NUM_VAL_TASKS=1500,
            NUM_TEST_TASKS=20000,
            # data information
            SQUARE_JITTER=True,
            # for meta-test
            # [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            TEST_SPATIAL_MODE=1,
            TEST_AS_VAL=False,
            UNIFIED_EVAL=True,
            CENTER_CROP_MULTI_VIEW=False,
        ),
        TEST=dict(
            NUM_RUNS=1,
        ),
    ),
    BN=dict(
        USE_PRECISE_STATS=False,
        NUM_BATCHES_PRECISE=200,
    ),
    TRAIN=dict(
        DATASET="minissv2",
        BATCH_SIZE=16,  # 64
        EVAL_PERIOD=5,
        CHECKPOINT_PERIOD=5,
        CHECKPOINT_FILE_PATH=osp.join(
            pyaction_home,
            "model_zoo/R50_IN1K.pyth",
        ),
        CHECKPOINT_INFLATE=True,
        # For few-shot learning
        TEST_AS_VAL=False,
    ),
    SOLVER=dict(
        BASE_LR=0.001,
        LR_POLICY="steps_with_relative_lrs",
        STEPS=[0, 90, 120, 160],
        LRS=[1, 0.1, 0.01, 0.001],
        MAX_EPOCH=180,
        # Momentum.
        MOMENTUM=0,
        # Nesterov momentum.
        NESTEROV=False,
        # L2 regularization.
        WEIGHT_DECAY=0,
    ),
    TEST=dict(
        ENABLE=True,
        DATASET="minissv2",
        BATCH_SIZE=32,
        NUM_ENSEMBLE_VIEWS=1,
        NUM_SPATIAL_CROPS=1,
        START_EPOCH=-1,
        END_EPOCH=-1,
        # For few-shot learning
        # Eval Mode
        UNIFIED_EVAL=True,
        CENTER_CROP_MULTI_VIEW=False,
        # Splits to eval
        SPLITS=["test"],  # {"train", "test"}
    ),
    DATA=dict(
        PATH_TO_DATA_DIR=osp.join(pyaction_home, "data/minissv2"),
        NUM_FRAMES=8,
        SAMPLING_RATE=8,
        INPUT_CHANNEL_NUM=[3],
        INV_UNIFORM_SAMPLE=True,
        RANDOM_FLIP=False,
        REVERSE_INPUT_CHANNEL=True,
        TEST_SCALE_SIZE=256,
        TEST_CROP_SIZE=224,
        # For few-shot learning
        SQUARE_JITTER=True,
    ),
    MODEL=dict(
        ARCH="c2d_nopool",
        NUM_CLASSES=64,
    ),
    RESNET=dict(
        ZERO_INIT_FINAL_BN=True,
        DEPTH=50,
        NUM_BLOCK_TEMP_KERNEL=[[3], [4], [6], [3]],
        # for few-shot
        GET_FEATURE=True,
        FEATURE_DIM=64,  # raw feature dim, added for few-shot
    ),
    DATA_LOADER=dict(
        NUM_WORKERS=4,
        PIN_MEMORY=True,
        # ENABLE_MULTI_THREAD_DECODE=True
    ),
    NUM_GPUS=4,
    NUM_SHARDS=1,
    RNG_SEED=2147,
    DIST_MULTIPROCESS=True,
)


class MetaSSv2Config(BaseConfig):
    def __init__(self):
        super(MetaSSv2Config, self).__init__()
        self._register_configuration(_config_dict)


config = MetaSSv2Config()
