# Installation Struction
## 0. Meta Learning with PyAction

```
# Use 4-gpus as default

# Train:
pyaction_meta_run

# Test with latest model
pyaction_meta_run TRAIN.ENABLE False

# Test with a specific file
pyaction_meta_run TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH /path-to-checkpoint-files/

# Test with an interval
pyaction_meta_run TRAIN.ENABLE False TEST.START_EPOCH 50 TEST.END_EPOCH 81
```

## 1. Enverionment Setup
### Conda Environment
```
conda create --name pytorch1.4 python=3.7
conda activate pytorch1.4
conda install pytorch=1.4.0 cudatoolkit=10.0 torchvision -c pytorch
conda install av -c conda-forge

pip install opencv-python scikit-learn gpustat
pip install -U cython pre-commit easydict colorama simplejson
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

```
### System Environment

#### Put the following into `pyactionenv.sh`
- P40 Cluster

You need to build gcc-5.2.0 first.
```
source activate pytorch1.4
# Folder to store the models and logs
export PYACTION_OUTPUT='/public/sist/home/hexm/Models/pyaction'
export PYACTION_HOME='/public/sist/home/hexm/Projects/pyaction'

#gcc 5.2.0
export PATH=~/local/bin:$PATH
export LD_LIBRARY_PATH=~/local/lib64:$LD_LIBRARY_PATH

# cuda 10.0
export PATH=/public/software/compiler/cuda/7/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/public/software/compiler/cuda/7/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- AI Cluster
```
source activate pytorch1.4
# Folder to store the models and logs
export PYACTION_OUTPUT='/public/sist/home/hexm/Models/pyaction'

```
#### Add `alias` in the `.bashrc`
```
alias pyactionenv="source /public/sist/home/hexm/conda_envs/pyaction.sh"
```

## 2. Project Build
You need to build the project in the cuda avaliable environment
```
# enter the codna environment
pyactionenv
# build the essential libs and scripts
cd pyaction
python setup.py build develop
```

## 3. Dataset Preparation
Prepare the dataset for training and evaluation in [DATASET.md](DATASET.md)

## 4. Project Training
1. Clone the workspace and link to the current folder
```bash
# at the root of the pyaction project
git clone https://github.com/tonysy/PyAction_Workspace workspace
```

2. We use folder-based develop pipeline. 

We only need to focus the model design within the project folder.
All experiments are stored into the `workspace`.

- Single Node

```
cd workspace/kinetics/c2d.kinetivs400.8x8.res50
# train with 4 GPUS as default
pyaction_run

# only test with 4 GPUS as default
pyaction_run TRAIN.ENABLE False
```

- Distributed train
```bash
# Node-1
pyaction_run --shard_id 0 --num_shards 2 --init_method tcp://gnode22:9999
# Node-2
pyaction_run --shard_id 1 --num_shards 2 --init_method tcp://gnode22:9999
```

## 5. Develop

### Code Style Check
You need to excute the following to allow the code style check before each `git commit`

```
pre-commit install
```