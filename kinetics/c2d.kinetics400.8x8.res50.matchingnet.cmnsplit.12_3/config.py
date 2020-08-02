import os
import os.path as osp

import pyaction
from pyaction.configs.base_config import BaseConfig

_config_dict = dict(
    BN=dict(
        USE_PRECISE_STATS=True, NUM_BATCHES_PRECISE=200, MOMENTUM=0.1, WEIGHT_DECAY=0.0,
    ),
    TRAIN=dict(
        DATASET="Kineticsnshot", 
        BATCH_SIZE=8, #64
        EVAL_PERIOD=10, 
        CHECKPOINT_PERIOD=10,
        # CHECKPOINT_FILE_PATH=osp.join(
        #     "/",
        #     *osp.realpath(pyaction.__file__).split("/")[:-2],
        #     "model_zoo/R50_IN1K.pyth",
        # ),
        CHECKPOINT_INFLATE=True,
    ), 
    DATA=dict(
        PATH_TO_DATA_DIR=osp.join(
            "/", *osp.realpath(pyaction.__file__).split("/")[:-2], "data/kineticsnshot/cmn"
        ),
        NUM_FRAMES=8,
        SAMPLING_RATE=8,
        INPUT_CHANNEL_NUM=[3],
    ),
    # SLOWFAST=dict(
    #     # Corresponds to the inverse of the channel reduction ratio, $\beta$ between
    #     # the Slow and Fast pathways.
    #     BETA_INV = 8,
    #     # Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
    #     # Fast pathways.
    #     ALPHA=4,
    #     # Ratio of channel dimensions between the Slow and Fast pathways.
    #     FUSION_CONV_CHANNEL_RATIO = 2,
    #     # Kernel dimension used for fusing information from Fast pathway to Slow
    #     # pathway.
    #     FUSION_KERNEL_SZ = 7
    # ),
    RESNET=dict(
        ZERO_INIT_FINAL_BN=True, DEPTH=50, NUM_BLOCK_TEMP_KERNEL=[[3], [4], [6], [3]],
        GET_FEATURE=True,
        FEATURE_DIM=64, # raw feature dim, added for few-shot
    ),
    # SOLVER=dict(BASE_LR=0.1, LR_POLICY="cosine", MAX_EPOCH=1000, WARMUP_EPOCHS=34,),  # without Imagetnet
    SOLVER=dict(
        BASE_LR=0.0001,
        LR_POLICY="steps_with_relative_lrs",
        STEPS=[0, 29, 59, 89],
        LRS=[1, 1, 1, 1],
        MAX_EPOCH=5000, # 120!
    ),
    MODEL=dict(ARCH="c2d", NUM_CLASSES=400,),
    TEST=dict(ENABLE=True, DATASET="Kineticsnshot", BATCH_SIZE=64),
    DATA_LOADER=dict(
        NUM_WORKERS=8,
        PIN_MEMORY=True,
        # ENABLE_MULTI_THREAD_DECODE=True
    ),
    DIST_MULTIPROCESS=True,
    NUM_GPUS=4, ############################
    NUM_SHARDS=1,
    RNG_SEED=0,
    OUTPUT_DIR=osp.join(
        os.getenv("PYACTION_OUTPUT"),
        "model_logs",
        *osp.realpath(__file__).split("/")[-3:-1],
    ),
    FEW_SHOT=dict(
        EPOCH_LEN=1500,
        CLASSES_PER_SET=5,
        SAMPLES_PER_CLASS=1,
        FCE=False,  #######################
    )
)


class KineticsConfig(BaseConfig):
    def __init__(self):
        super(KineticsConfig, self).__init__()
        self._register_configuration(_config_dict)


config = KineticsConfig()
