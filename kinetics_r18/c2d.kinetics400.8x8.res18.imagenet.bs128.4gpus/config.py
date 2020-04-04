import os
import os.path as osp

import pyaction
from pyaction.configs.base_config import BaseConfig

_config_dict = dict(
    BN=dict(
        USE_PRECISE_STATS=True, NUM_BATCHES_PRECISE=200, MOMENTUM=0.1, WEIGHT_DECAY=0.0,
    ),
    TRAIN=dict(
        DATASET="kinetics", 
        # BATCH_SIZE=64, 
        BATCH_SIZE=128, 
        EVAL_PERIOD=10, 
        CHECKPOINT_PERIOD=1,
        CHECKPOINT_FILE_PATH=osp.join(
            "/",
            *osp.realpath(pyaction.__file__).split("/")[:-2],
            "model_zoo/R18_IN1K_SY.pyth",
        ),
        CHECKPOINT_INFLATE=True,),
    DATA=dict(
        PATH_TO_DATA_DIR=osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/kinetics"
        ),
        NUM_FRAMES=8,
        SAMPLING_RATE=8,
        INPUT_CHANNEL_NUM=[3],
    ),
    RESNET=dict(
        TRANS_FUNC='basic_transform',
        ZERO_INIT_FINAL_BN=True, 
        DEPTH=18, 
        NUM_BLOCK_TEMP_KERNEL=[[2], [2], [2], [2]],
    ),
    SOLVER=dict(
        BASE_LR=0.02,#0.01,
        LR_POLICY="steps_with_relative_lrs",
        STEPS=[0, 44, 88, 118],
        LRS=[1, 0.1, 0.01, 0.001],
        MAX_EPOCH=118,
        ),
    MODEL=dict(ARCH="c2d", NUM_CLASSES=400,),
    TEST=dict(ENABLE=True, DATASET="kinetics", BATCH_SIZE=64),
    DATA_LOADER=dict(
        NUM_WORKERS=8,
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
