# Dual V-Learning

Official code base for Dual RL: Dual RL: Unification and New Methods for Reinforcement and Imitation Learning



## How to run the code

### Install dependencies

Create an empty conda environment and follow the commands below.

```bash
conda create -n dvl python=3.9

conda install -c conda-forge cudnn

pip install --upgrade pip

# Install 1 of the below jax versions depending on your CUDA version
## 1. CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

## 2. CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


pip install -r requirements.txt

```

### Example training code

Locomotion
```bash
python train_offline.py --env_name=halfcheetah-medium-expert-v2 --f=chi-square --config=configs/mujoco_config.py --max_clip=5 --sample_random_times=1 --temp=1
```

AntMaze
```bash
python train_offline.py --env_name=antmaze-large-play-v0 --f=total-variation --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000  --max_clip=5  --temp=0.8
```

Kitchen and Adroit
```bash
python train_offline.py --env_name=pen-human-v0  --f=reverse-KL --config=configs/kitchen_config.py --max_clip=5 --sample_random_times=1 --temp=8
```



## Acknowledgement and Reference

This code base heavily builds upon the following code bases: [Extreme Q-learning](https://github.com/Div99/XQL) and [Implicit Q-Learning](https://github.com/ikostrikov/implicit_q_learning).
