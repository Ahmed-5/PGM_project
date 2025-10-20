# First time: login to wandb
wandb login

# Train with wandb logging
python train.py --config exponential_decay --logger wandb --wandb-project my-project --wandb-entity my-username

# Train with custom experiment name
python train.py --config learnable --logger wandb --experiment-name learnable_v2
