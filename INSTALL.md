# Installation Struction

## 1. Enverionment Setup
### Conda Environment
```
conda create --name pytorch1.4 python=3.7
conda activate pytorch1.4
conda install pytorch=1.4.0 cudatoolkit=10.0 torchvision -c pytorch

pip install -U cython
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

## 3. Project Training

We use folder-based develop pipeline. We only need to focus the model design within the project folder.
All experiments are stored into the `workspace`.
```
cd workspace/kinetics/c2d.kinetivs400.8x8.res50
# train with 4 GPUS as default
pyaction_run

# only test with 4 GPUS as default
pyaction_run TRAIN.ENABLE False
```

## 4. Develop

### Code Style Check
You need to excute the following to allow the code style check before each `git commit`

```
pre-commit install
```