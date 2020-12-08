import os
import os.path as osp

import pyaction
from pyaction.configs.base_config import BaseConfig

pyaction_home = os.path.dirname(pyaction.__path__[0])

_config_dict = dict(
    BN=dict(
        USE_PRECISE_STATS=True,
        NUM_BATCHES_PRECISE=200,
        MOMENTUM=0.1,
        WEIGHT_DECAY=0.0,
    ),
    TRAIN=dict(
        DATASET="kinetics",
        BATCH_SIZE=64,
        EVAL_PERIOD=10,
        CHECKPOINT_PERIOD=5,
    ),
    DATA=dict(
        PATH_TO_DATA_DIR=osp.join(pyaction_home, "data/kinetics"),
        NUM_FRAMES=8,
        SAMPLING_RATE=8,
        INPUT_CHANNEL_NUM=[3],
    ),
    RESNET=dict(
        ZERO_INIT_FINAL_BN=True,
        DEPTH=50,
        NUM_BLOCK_TEMP_KERNEL=[[3], [4], [6], [3]],
    ),
    MODEL=dict(
        ARCH="c2d",
        NUM_CLASSES=400,
    ),
    TEST=dict(ENABLE=True, DATASET="kinetics", BATCH_SIZE=64),
    DATA_LOADER=dict(
        NUM_WORKERS=4,
        PIN_MEMORY=True,
        # ENABLE_MULTI_THREAD_DECODE=True
    ),
    DIST_MULTIPROCESS=True,
    NUM_GPUS=4,
    NUM_SHARDS=1,
    RNG_SEED=0,
)


class KineticsConfig(BaseConfig):
    def __init__(self):
        super(KineticsConfig, self).__init__()
        self._register_configuration(_config_dict)


config = KineticsConfig()
