# Train with TensorBoard logging
python train.py --config exponential_decay --logger tensorboard

# View logs in another terminal
tensorboard --logdir=./runs
