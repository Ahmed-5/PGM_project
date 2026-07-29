import argparse
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Type, Any
import config as cfg_module
from train import train


def str2bool(value) -> bool:
    """argparse type for booleans: true/1/t/yes/y and false/0/f/no/n."""
    if isinstance(value, bool):
        return value
    v = str(value).lower()
    if v in ('true', '1', 't', 'yes', 'y'):
        return True
    if v in ('false', '0', 'f', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'")


def add_arguments_from_dataclass(parser: argparse.ArgumentParser, 
                               dtype: Type, 
                               prefix: str = "") -> None:
    """Recursively add arguments from a dataclass."""
    for field in fields(dtype):
        # Handle nested dataclasses (e.g., config.model -> --model.layers)
        if is_dataclass(field.type):
            add_arguments_from_dataclass(
                parser, 
                field.type, 
                prefix=f"{prefix}{field.name}."
            )
            continue

        # Create argument name (e.g., --training.learning_rate)
        # arg_name = f"--{prefix}{field.name}".replace("_", "-")
        arg_name = f"--{prefix}{field.name}"
        
        # Determine type for argparse
        field_type = field.type
        
        # simple heuristic for types; can be expanded for Lists/Dicts
        if hasattr(field_type, "__origin__"):  # Handle Optional[int], List[str] etc
            origin = field_type.__origin__
            if origin is list:
                # Naive handling for lists: expects space separated
                # Better to use json.loads for complex types via a custom action
                parser.add_argument(arg_name, nargs='+', help=f"{prefix}{field.name}")
                continue
            elif origin is dict:
                 # For dicts, use a JSON string parsing strategy
                 parser.add_argument(arg_name, type=str, help=f"JSON string for {prefix}{field.name}")
                 continue
            
        # Add standard argument
        # Note: We don't set defaults here to avoid overriding the dataclass defaults
        # unless the user explicitly provides the flag.
        # Booleans need str2bool: type=bool would make "--flag false" truthy.
        if field_type is bool:
            parser.add_argument(arg_name, type=str2bool,
                                required=False, help=f"Override for {prefix}{field.name}")
        else:
            parser.add_argument(arg_name, type=field_type if field_type in [int, float, str] else str, 
                                required=False, help=f"Override for {prefix}{field.name}")

def apply_args_to_config(args: argparse.Namespace, config: cfg_module.ExperimentConfig):
    """Recursively update config with parsed args."""
    args_dict = vars(args)
    
    for key, value in args_dict.items():
        if value is None:
            continue
            
        # Navigate to the correct nested object
        parts = key.split(".")
        target = config
        
        # Go deep
        for part in parts[:-1]:
            target = getattr(target, part)
            
        # Set value
        field_name = parts[-1]
        
        # Handle special types if necessary (lists, dicts)
        current_val = getattr(target, field_name)
        
        # If the target is a list: accept either space-separated tokens
        # (nargs='+') or a single Python/JSON list literal string, e.g.
        # "['so3', 'translation']".
        if isinstance(current_val, list):
            import ast
            parsed = value
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                try:
                    literal = ast.literal_eval(value[0])
                    if isinstance(literal, (list, tuple)):
                        parsed = list(literal)
                except (ValueError, SyntaxError):
                    parsed = value
            elif isinstance(value, str):
                try:
                    literal = ast.literal_eval(value)
                    parsed = list(literal) if isinstance(literal, (list, tuple)) else [value]
                except (ValueError, SyntaxError):
                    parsed = [value]
            setattr(target, field_name, parsed)
        # If the target is a dict, parse a Python/JSON dict literal string,
        # e.g. "{'so3': 0.5}" (single quotes allowed via ast.literal_eval).
        elif isinstance(current_val, dict) and isinstance(value, str):
             import ast, json
             try:
                 setattr(target, field_name, ast.literal_eval(value))
             except (ValueError, SyntaxError):
                 try:
                     setattr(target, field_name, json.loads(value))
                 except Exception:
                     print(f"Warning: Could not parse dict for {key}={value}")
        else:
            # Simple type casting
            target_type = type(current_val)
            try:
                if target_type is bool:
                    # argparse already produced a bool via str2bool; this also
                    # covers values set programmatically.
                    setattr(target, field_name, str2bool(value))
                else:
                    setattr(target, field_name, target_type(value))
            except (ValueError, argparse.ArgumentTypeError):
                print(f"Warning: Could not cast {key}={value} to {target_type}")

def get_args_parser():
    parser = argparse.ArgumentParser(description="Universal Training CLI")
    add_arguments_from_dataclass(parser, cfg_module.ExperimentConfig)
    return parser


def _remove_dir_if_empty(path: str) -> None:
    """Remove a placeholder directory (only if empty); ignore failures."""
    try:
        Path(path).rmdir()
    except OSError:
        pass


def build_config(args: argparse.Namespace) -> cfg_module.ExperimentConfig:
    """Build a fully-overridden, validated ExperimentConfig from CLI args.

    The initial ExperimentConfig() construction finalizes against the DEFAULTS
    (creating ./checkpoints/default_<ts> and ./outputs/default_<ts>), so after
    applying overrides we recompute the run directories from the final
    experiment_name (unless explicitly overridden), drop the empty placeholder
    dirs, and re-run finalize() so validation sees the final config.
    """
    # 1. Start with default config
    config = cfg_module.ExperimentConfig()
    default_checkpoint_dir = config.checkpoint_dir
    default_output_dir = config.output_dir

    # 2. Apply overrides
    apply_args_to_config(args, config)

    # 3. Recompute run dirs from the final experiment_name (explicit
    #    --checkpoint_dir/--output_dir flags win), then re-finalize.
    if args.checkpoint_dir is None:
        config.checkpoint_dir = f'./checkpoints/{config.experiment_name}_{config.timestamp}'
    if args.output_dir is None:
        config.output_dir = f'./outputs/{config.experiment_name}_{config.timestamp}'

    if config.checkpoint_dir != default_checkpoint_dir:
        _remove_dir_if_empty(default_checkpoint_dir)
    if config.output_dir != default_output_dir:
        _remove_dir_if_empty(default_output_dir)

    config.finalize()
    return config


# Usage in train.py
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    config = build_config(args)
    train(config)
