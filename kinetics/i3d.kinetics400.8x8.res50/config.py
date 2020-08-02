import os
import os.path as osp

import pyaction
from pyaction.configs.base_config import BaseConfig

_config_dict = dict(
    BN=dict(
        USE_PRECISE_STATS=True, NUM_BATCHES_PRECISE=200, MOMENTUM=0.1, WEIGHT_DECAY=0.0,
    ),
    TRAIN=dict(DATASET="kinetics", BATCH_SIZE=64, EVAL_PERIOD=10, CHECKPOINT_PERIOD=1,),
    DATA=dict(
        PATH_TO_DATA_DIR=osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/kinetics"
        ),
        NUM_FRAMES=8,
        SAMPLING_RATE=8,
        INPUT_CHANNEL_NUM=[3],
    ),
    RESNET=dict(
        ZERO_INIT_FINAL_BN=True, DEPTH=50, NUM_BLOCK_TEMP_KERNEL=[[3], [4], [6], [3]],
    ),
    SOLVER=dict(BASE_LR=0.1, LR_POLICY="cosine", MAX_EPOCH=196, WARMUP_EPOCHS=34,),
    MODEL=dict(ARCH="i3d", NUM_CLASSES=400,),
    TEST=dict(ENABLE=True, DATASET="kinetics", BATCH_SIZE=64),
    DATA_LOADER=dict(NUM_WORKERS=8, PIN_MEMORY=True,),
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
