# PySlowFast Model Zoo and Baselines

## Kinetics

We provided original pretrained models from Caffe2 on heavy models (testing Caffe2 pretrained model in PyTorch might have a small different in performance):

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