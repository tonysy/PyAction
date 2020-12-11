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
        DATASET="ssv2",
        BATCH_SIZE=16,
        EVAL_PERIOD=2,
        CHECKPOINT_PERIOD=2,
        CHECKPOINT_FILE_PATH=osp.join(
            pyaction_home,
            "model_zoo/kinetics400/slowfast/SLOWFAST_8x8_R50.pkl",
        ),
        CHECKPOINT_TYPE="caffe2",
    ),
    DATA=dict(
        PATH_TO_DATA_DIR=osp.join(pyaction_home, "data/ssv2/"),
        PATH_PREFIX=osp.join(pyaction_home, "data/ssv2/frames"),
        NUM_FRAMES=8,
        SAMPLING_RATE=8,
        INPUT_CHANNEL_NUM=[3],
        INV_UNIFORM_SAMPLE=True,
        RANDOM_FLIP=False,
        REVERSE_INPUT_CHANNEL=True,
    ),
    MODEL=dict(
        ARCH="c2d",
        NUM_CLASSES=400,
    ),
    TEST=dict(
        ENABLE=True,
        DATASET="ssv2",
        BATCH_SIZE=16,
        NUM_ENSEMBLE_VIEWS=1,
        NUM_SPATIAL_CROPS=3,
    ),
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


class SSV2Config(BaseConfig):
    def __init__(self):
        super(SSV2Config, self).__init__()
        self._register_configuration(_config_dict)


config = SSV2Config()
