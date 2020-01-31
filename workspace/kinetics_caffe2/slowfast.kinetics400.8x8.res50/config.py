import os
import os.path as osp

import pyaction
from pyaction.configs.base_config import BaseConfig

_config_dict = dict(
    BN=dict(
        USE_PRECISE_STATS=True, NUM_BATCHES_PRECISE=200, MOMENTUM=0.1, WEIGHT_DECAY=0.0,
    ),
    TRAIN=dict(
        ENABLE=False,
        DATASET="kinetics",
        BATCH_SIZE=64,
        EVAL_PERIOD=10,
        CHECKPOINT_PERIOD=1,
        CHECKPOINT_TYPE="caffe2",
    ),
    DATA=dict(
        PATH_TO_DATA_DIR=osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/kinetics"
        ),
        NUM_FRAMES=32,
        SAMPLING_RATE=2,
        INPUT_CHANNEL_NUM=[3, 3],
    ),
    SLOWFAST=dict(
        # Corresponds to the inverse of the channel reduction ratio, $\beta$ between
        # the Slow and Fast pathways.
        BETA_INV=8,
        # Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
        # Fast pathways.
        ALPHA=4,
        # Ratio of channel dimensions between the Slow and Fast pathways.
        FUSION_CONV_CHANNEL_RATIO=2,
        # Kernel dimension used for fusing information from Fast pathway to Slow
        # pathway.
        FUSION_KERNEL_SZ=7,
    ),
    RESNET=dict(
        ZERO_INIT_FINAL_BN=True,
        DEPTH=50,
        NUM_BLOCK_TEMP_KERNEL=[[3, 3], [4, 4], [6, 6], [3, 3]],
        SPATIAL_STRIDES=[[1, 1], [2, 2], [2, 2], [2, 2]],
        SPATIAL_DILATIONS=[[1, 1], [1, 1], [1, 1], [1, 1]],
    ),
    NONLOCAL=dict(
        LOCATION=[[[], []], [[], []], [[], []], [[], []]],
        GROUP=[[1, 1], [1, 1], [1, 1], [1, 1]],
        # INSTANTIATION='dot_product'
    ),
    SOLVER=dict(BASE_LR=0.1, LR_POLICY="cosine", MAX_EPOCH=196, WARMUP_EPOCHS=34,),
    MODEL=dict(ARCH="slowfast", NUM_CLASSES=400,),
    TEST=dict(
        ENABLE=True,
        DATASET="kinetics",
        BATCH_SIZE=64,
        CHECKPOINT_TYPE="caffe2",
        CHECKPOINT_FILE_PATH=osp.join(
            "/",
            *osp.realpath(pyaction.__file__).split("/")[:-2],
            "model_zoo/SLOWFAST_8x8_R50.pkl",
        ),
    ),
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