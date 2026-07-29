# Train with TensorBoard logging (valid presets: default, baseline, e3_equivariant, multi_symmetry)
python train.py --config baseline --logger tensorboard

# View logs in another terminal
tensorboard --logdir=./runs
