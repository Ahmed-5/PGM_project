# First time: login to wandb
wandb login

# Train with wandb logging (valid presets: default, baseline, e3_equivariant, multi_symmetry)
python train.py --config baseline --logger wandb --wandb-project my-project --wandb-entity my-username

# Train with custom experiment name
python train.py --config multi_symmetry --logger wandb --experiment-name multi_symmetry_v2
