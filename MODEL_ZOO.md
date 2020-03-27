# PySlowFast Model Zoo and Baselines

## Kinetics

We provided original pretrained models from Caffe2 on heavy models (testing Caffe2 pretrained model in PyTorch might have a small different in performance):

### Trained in PyAction
| architecture | depth |  pretrain |  frame length x sample rate | top1 |  top5  |  top1(Our) |  top5(Our)  | model | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- | ------------- |
| C2D | R50 | [ImageNet Res-50](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/R50_IN1K.pyth) | 8 x 8 | 72.1 | 89.9 | [`link`]() |workspace/kinetics/c2d.kinetics400.8x8.res50.imagenet|
| C2D NLN | R50 | [ImageNet Res-50](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/R50_IN1K.pyth) | 8 x 8 | 74.0 | 91.1 | [`link`]() |workspace/kinetics/c2d.nonlocal.kinetics400.8x8.res50.imagenet|
| I3D | R50 | [ImageNet Res-50](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/R50_IN1K.pyth) | 8 x 8 |  |  | [`link`]() |workspace/kinetics/i3d.kinetics400.8x8.res50.imagenet|
| I3D NLN | R50 | [ImageNet Res-50](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/R50_IN1K.pyth) | 8 x 8 | 74.2 | 91.4 | [`link`]() |workspace/kinetics/i3d.nonlocal.kinetics400.8x8.res50.imagenet|
* +NLN use 5 blocks in all experiments

### Provided in SlowFast
| architecture | depth |  pretrain |  frame length x sample rate | top1 |  top5  |  top1(Our) |  top5(Our)  | model | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- | ------------- |
| C2D | R50 | Train From Scratch | 8 x 8 | 67.2 | 87.8 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/C2D_NOPOOL_8x8_R50.pkl) | Kinetics/c2/C2D_NOPOOL_8x8_R50 |
| I3D | R50 | Train From Scratch | 8 x 8 | 73.5 | 90.8 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl) | Kinetics/c2/I3D_8x8_R50 |
| I3D NLN | R50 | Train From Scratch | 8 x 8 | 74.0 | 91.1 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_NLN_8x8_R50.pkl) | Kinetics/c2/I3D_NLN_8x8_R50 |
| SlowOnly | R50 | Train From Scratch | 4 x 16 | 72.7 | 90.3 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_4x16_R50.pkl) | Kinetics/c2/SLOWONLY_4x16_R50 |
| SlowOnly | R50 | Train From Scratch | 8 x 8 | 74.8 | 91.6 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_8x8_R50.pkl) | Kinetics/c2/SLOWONLY_8x8_R50 |
| SlowFast | R50 | Train From Scratch | 4 x 16 | 75.6 | 92.0 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_4x16_R50.pkl) | Kinetics/c2/SLOWFAST_4x16_R50 |
| SlowFast | R50 | Train From Scratch | 8 x 8 | 77.0 | 92.6 | 76.38 | 92.22 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl) | workspace/kinetics_caffe2/slowfast.kinetics400.8x8.res50 |
| SlowFast | R101 | Train From Scratch | 8 x 8 | 78.0 | 93.3 | [`link`](coming_soon) | Kinetics/c2/SLOWFAST_8x8_R101_101_101|
| SlowFast | R101 | Train From Scratch | 16 x 8 | 78.9 | 93.5 | [`link`](coming_soon) | Kinetics/c2/SLOWFAST_16x8_R101_50_50 |

## AVA
### Trained in PyAction
| architecture | depth | Pretrain Model |  frame length x sample rate  | MAP | AVA version | model | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |
| C2D      | R50 | [Kinetics 400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/C2D_8x8_R50.pkl) | 4 x 16 |  13.9    | 2.2 | | workspace/ava/c2d.ava.8x8.res50.short

### Provided in SlowFast
| architecture | depth | Pretrain Model |  frame length x sample rate  | MAP | AVA version | model | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |
| SlowOnly | R50 | [Kinetics 400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/C2D_8x8_R50.pkl) | 4 x 16 | 19.5 | 2.2 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/C2D_8x8_R50.pkl) |
| SlowFast | R101 | [Kinetics 600](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_32x2_R101_50_50_v2.1.pkl) | 8 x 8 | 28.2 | 2.1 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_32x2_R101_50_50_v2.1.pkl) | workspace/ava_caffe2/slowfast.32x2.res101.50.50.v2.1|
| SlowFast | R101 | [Kinetics 600](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_32x2_R101_50_50.pkl) | 8 x 8 | 29.1 | 2.2 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_32x2_R101_50_50.pkl) |workspace/ava_caffe2/slowfast.32x2.res101.50.50.v2.2 |
| SlowFast | R101 | [Kinetics 600](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_64x2_R101_50_50.pkl) | 16 x 8 | 29.4 | 2.2 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_64x2_R101_50_50.pkl) |workspace/ava_caffe2/slowfast.64x2.res101.50.50|

