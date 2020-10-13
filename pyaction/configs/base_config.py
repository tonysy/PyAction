import collections
import logging
import os
import pprint
import re
from ast import literal_eval
import six
from colorama import Back, Fore
from easydict import EasyDict as edict

from pyaction.utils.config_helper import diff_dict, find_key, highlight

# python 3.8+ compatibility
try:
    collectionsAbc = collections.abc
except ImportError:
    collectionsAbc = collections


_config_dict = dict(
    # For PyAction
    BN=dict(
        # BN epsilon.
        EPSILON=1e-5,
        # BN momentum.
        MOMENTUM=0.1,
        # Precise BN stats.
        USE_PRECISE_STATS=False,
        # Number of samples use to compute precise bn.
        NUM_BATCHES_PRECISE=200,
        # Weight decay value that applies on BN.
        WEIGHT_DECAY=0.0,
    ),
    TRAIN=dict(
        # If True Train the model, else skip training.
        ENABLE=True,
        # Dataset.
        DATASET="kinetics",
        # Total mini-batch size.
        BATCH_SIZE=64,
        # Evaluate model on test data every eval period epochs.
        EVAL_PERIOD=1,
        # Save model checkpoint every checkpoint period epochs.
        CHECKPOINT_PERIOD=1,
        # Resume training from the latest checkpoint in the output directory.
        AUTO_RESUME=True,
        # Path to the checkpoint to load the initial weight.
        CHECKPOINT_FILE_PATH="",
        # Checkpoint types include `caffe2` or `pytorch`.
        CHECKPOINT_TYPE="pytorch",
        # If True, perform inflation when loading checkpoint.
        CHECKPOINT_INFLATE=False,
    ),
    TEST=dict(
        # If True test the model, else skip the testing.
        ENABLE=True,
        # Dataset for testing.
        DATASET="kinetics",
        # Total mini-batch size
        BATCH_SIZE=8,
        # Path to the checkpoint to load the initial weight.
        CHECKPOINT_FILE_PATH="",
        # Number of clips to sample from a video uniformly for aggregating the
        # prediction results.
        NUM_ENSEMBLE_VIEWS=10,
        # Number of crops to sample from a frame spatially for aggregating the
        # prediction results.
        NUM_SPATIAL_CROPS=3,
        # Checkpoint types include `caffe2` or `pytorch`.
        CHECKPOINT_TYPE="pytorch",
    ),
    RESNET=dict(
        # Transformation function.
        TRANS_FUNC="bottleneck_transform",
        # Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
        NUM_GROUPS=1,
        # Width of each group (64 -> ResNet; 4 -> ResNeXt).
        WIDTH_PER_GROUP=64,
        # Apply relu in a inplace manner.
        INPLACE_RELU=True,
        # Apply stride to 1x1 conv.
        STRIDE_1X1=False,
        #  If true, initialize the gamma of the final BN of each block to zero.
        ZERO_INIT_FINAL_BN=False,
        # Number of weight layers.
        DEPTH=50,
        # If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
        # kernel of 1 for the rest of the blocks.
        NUM_BLOCK_TEMP_KERNEL=[[3], [4], [6], [3]],
        # Size of stride on different res stages.
        SPATIAL_STRIDES=[[1], [2], [2], [2]],
        # Size of dilation on different res stages.
        SPATIAL_DILATIONS=[[1], [1], [1], [1]],
    ),
    MODEL=dict(
        # Model architecture.
        ARCH="slowfast",
        # The number of classes to predict for the model.
        NUM_CLASSES=400,
        # Loss function.
        LOSS_FUNC="cross_entropy",
        # Model architectures that has one single pathway.
        SINGLE_PATHWAY_ARCH=["c2d", "i3d", "slowonly", "c2d_nopool", "i3d_nopool"],
        # Model architectures that has multiple pathways.
        MULTI_PATHWAY_ARCH=["slowfast"],
        # Dropout rate before final projection in the backbone.
        DROPOUT_RATE=0.5,
        # The std to initialize the fc layer(s).
        FC_INIT_STD=0.01,
    ),
    NONLOCAL=dict(
        # Index of each stage and block to add nonlocal layers.
        LOCATION=[[[]], [[]], [[]], [[]]],
        # Number of group for nonlocal for each stage.
        GROUP=[[1], [1], [1], [1]],
        # Instatiation to use for non-local layer.
        INSTANTIATION="dot_product",
        # Size of pooling layers used in Non-Local.
        POOL=[
            # Res2
            [[1, 2, 2], [1, 2, 2]],
            # Res3
            [[1, 2, 2], [1, 2, 2]],
            # Res4
            [[1, 2, 2], [1, 2, 2]],
            # Res5
            [[1, 2, 2], [1, 2, 2]],
        ],
    ),
    DETECTION=dict(
        # Whether enable video detection.
        ENABLE=False
    ),
    DATA=dict(
        # The path to the data directory.
        PATH_TO_DATA_DIR="",
        # Video path prefix if any.
        PATH_PREFIX="",
        # The spatial crop size of the input clip.
        CROP_SIZE=224,
        # The number of frames of the input clip.
        NUM_FRAMES=8,
        # The video sampling rate of the input clip.
        SAMPLING_RATE=8,
        # The mean value of the video raw pixels across the R G B channels.
        MEAN=[0.45, 0.45, 0.45],
        # List of input frame channel dimensions.
        INPUT_CHANNEL_NUM=[3, 3],
        # The std value of the video raw pixels across the R G B channels.
        STD=[0.225, 0.225, 0.225],
        # The spatial augmentation jitter scales for training.
        TRAIN_JITTER_SCALES=[256, 320],
        # The spatial crop size for training.
        TRAIN_CROP_SIZE=224,
        # The spatial crop size for testing.
        TEST_CROP_SIZE=256,
    ),
    DATA_LOADER=dict(
        # Number of data loader workers per training process.
        NUM_WORKERS=2,
        # Load data to pinned host memory.
        PIN_MEMORY=True,
        # Enable multi thread decoding.
        ENABLE_MULTI_THREAD_DECODE=False,
    ),
    SOLVER=dict(
        # Base learning rate.
        BASE_LR=0.1,
        # Learning rate policy (see utils/lr_policy.py for options and examples).
        LR_POLICY="cosine",
        # Exponential decay factor.
        GAMMA=0.1,
        # Step size for 'exp' and 'cos' policies (in epochs).
        STEP_SIZE=1,
        # Steps for 'steps_' policies (in epochs).
        STEPS=[],
        # Learning rates for 'steps_' policies.
        LRS=[],
        # Maximal number of epochs.
        MAX_EPOCH=300,
        # Momentum.
        MOMENTUM=0.9,
        # Momentum dampening.
        DAMPENING=0.0,
        # Nesterov momentum.
        NESTEROV=True,
        # L2 regularization.
        WEIGHT_DECAY=1e-4,
        # Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
        WARMUP_FACTOR=0.1,
        # Gradually warm up the SOLVER.BASE_LR over this number of epochs.
        WARMUP_EPOCHS=0,
        # The start learning rate of the warm up.
        WARMUP_START_LR=0.01,
        # Optimization method.
        OPTIMIZING_METHOD="sgd",
    ),
    # Number of GPUs to use (applies to both training and testing).
    NUM_GPUS=1,
    # Number of machine to use for the job.
    NUM_SHARDS=1,
    # The index of the current machine.
    SHARD_ID=0,
    # Output basedir.
    OUTPUT_DIR="./tmp",
    # Note that non-determinism may still be present due to non-deterministic
    # operator implementations in GPU operator libraries.
    RNG_SEED=1,
    # Log period in iters.
    LOG_PERIOD=10,
    # Distributed backend.
    DIST_BACKEND="nccl",
)


def _assert_with_logging(cond, msg):
    logger = logging.getLogger(__name__)
    if not cond:
        logger.debug(msg)
    assert cond, msg


def update(d, u):
    for k, v in six.iteritems(u):
        dv = d.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            d[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            d[k] = update(dv, v)
        else:
            d[k] = v
    return d


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


class BaseConfig(object):
    def __init__(self):
        # self._register_configuration(self.check_update(_config_dict))
        self._register_configuration(_config_dict)

    def _register_configuration(self, config):
        for k, v in config.items():
            if hasattr(self, k):
                if isinstance(v, dict):
                    setattr(self, k, update(getattr(self, k), v))
                else:
                    setattr(self, k, v)
            elif isinstance(v, dict):
                setattr(self, k, edict(v))
            else:
                setattr(self, k, v)

    def delete_elems(self, cfg, elems):
        for ele in elems:
            cfg.pop(ele)
        return cfg

    def check_update(self, cfg):
        input_cfg = cfg["INPUT"]
        contains = [
            k in input_cfg.keys()
            for k in [
                "MIN_SIZE_TRAIN",
                "MIN_SIZE_TRAIN_SAMPLING",
                "MAX_SIZE_TRAIN",
                "MIN_SIZE_TEST",
                "MAX_SIZE_TEST",
            ]
        ]
        train_transforms = [t[0] for t in input_cfg["AUG"]["TRAIN_PIPELINES"]]
        test_transforms = [t[0] for t in input_cfg["AUG"]["TEST_PIPELINES"]]
        if any(contains):
            train_tup = (
                "ResizeShortestEdge",
                dict(
                    short_edge_length=input_cfg["MIN_SIZE_TRAIN"],
                    max_size=input_cfg["MAX_SIZE_TRAIN"],
                    sample_style=input_cfg["MIN_SIZE_TRAIN_SAMPLING"],
                ),
            )
            test_tup = (
                "ResizeShortestEdge",
                dict(
                    short_edge_length=input_cfg["MIN_SIZE_TEST"],
                    max_size=input_cfg["MAX_SIZE_TEST"],
                    sample_style="choice",
                ),
            )

            if "ResizeShortestEdge" in train_transforms:
                idx = train_transforms.index("ResizeShortestEdge")
                input_cfg["AUG"]["TRAIN_PIPELINES"][idx] = train_tup
            else:
                input_cfg["AUG"]["TRAIN_PIPELINES"].insert(0, train_tup)

            if "ResizeShortestEdge" in test_transforms:
                idx = test_transforms.index("ResizeShortestEdge")
                input_cfg["AUG"]["TEST_PIPELINES"][idx] = test_tup
            else:
                input_cfg["AUG"]["TEST_PIPELINES"].insert(0, test_tup)

            for elem in [
                "MIN_SIZE_TRAIN",
                "MAX_SIZE_TRAIN",
                "MIN_SIZE_TRAIN_SAMPLING",
                "MIN_SIZE_TEST",
                "MAX_SIZE_TEST",
            ]:
                cfg["INPUT"].pop(elem)

        return cfg

    def merge_from_list(self, cfg_list):
        """Merge config (keys, values) in a list (e.g., from command line) into
        this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            ),
        )
        # root = self
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(
                    hasattr(d, subkey), "Non-existent key: {}".format(full_key)
                )
                d = getattr(d, subkey)
            subkey = key_list[-1]
            _assert_with_logging(
                hasattr(d, subkey), "Non-existent key: {}".format(full_key)
            )
            value = self._decode_cfg_value(v)
            value = _check_and_coerce_cfg_value_type(
                value, getattr(d, subkey), subkey, full_key
            )
            setattr(d, subkey, value)

    def link_log(self, link_name="log"):
        if os.path.islink(link_name) and os.readlink(link_name) != self.OUTPUT_DIR:
            os.system("rm " + link_name)
        if not os.path.exists(link_name):
            cmd = "ln -s {} {}".format(self.OUTPUT_DIR, link_name)
            os.system(cmd)

    @classmethod
    def _decode_cfg_value(cls, value):
        """
        Decodes a raw config value (e.g., from a yaml config files or command
        line argument) into a Python object.
        If the value is a dict, it will be interpreted as a new CfgNode.
        If the value is a str, it will be evaluated as literals.
        Otherwise it is returned as-is.
        """
        # Configs parsed from raw yaml will contain dictionary keys that need to be
        # converted to CfgNode objects
        if isinstance(value, dict):
            return cls(value)
        # All remaining processing is only applied to strings
        if not isinstance(value, str):
            return value
        # Try to interpret `value` as a:
        #   string, number, tuple, list, dict, boolean, or None
        try:
            value = literal_eval(value)
        # The following two excepts allow v to pass through when it represents a
        # string.
        #
        # Longer explanation:
        # The type of v is always a string (before calling literal_eval), but
        # sometimes it *represents* a string and other times a data structure, like
        # a list. In the case that v represents a string, what we got back from the
        # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
        # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
        # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
        # will raise a SyntaxError.
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value

    def _get_param_list(self) -> list:
        param_list = [
            name
            for name in self.__dir__()
            if name[:2] != "__" and not callable(getattr(self, name))
        ]
        return param_list

    def diff(self, config) -> dict:
        """
        """
        assert isinstance(config, BaseConfig), "config is not a subclass of BaseConfig"
        diff_result = {}
        self_param_list = self._get_param_list()
        conf_param_list = config._get_param_list()
        for param in self_param_list:
            if param not in conf_param_list:
                diff_result[param] = getattr(self, param)
            else:
                self_val, conf_val = (
                    getattr(self, param),
                    getattr(config, param),
                )
                if self_val != conf_val:
                    if isinstance(self_val, dict):
                        diff_result[param] = diff_dict(self_val, conf_val)
                    else:
                        diff_result[param] = self_val
        return diff_result

    def show_diff(self, config):
        return pprint.pformat(edict(self.diff(config)))

    def find(self, key: str, show=True, color=Fore.BLACK + Back.YELLOW) -> dict:
        key = key.upper()
        find_result = {}
        param_list = self._get_param_list()
        for param in param_list:
            param_value = getattr(self, param)
            if re.search(key, param):
                find_result[param] = param_value
            elif isinstance(param_value, dict):
                find_res = find_key(param_value, key)
                if find_res:
                    find_result[param] = find_res
        if not show:
            return find_result
        else:
            pformat_str = pprint.pformat(edict(find_result))
            print(highlight(key, pformat_str, color))

    def __repr__(self):
        param_dict = edict(
            {param: getattr(self, param) for param in self._get_param_list()}
        )
        return pprint.pformat(param_dict)


config = BaseConfig()
