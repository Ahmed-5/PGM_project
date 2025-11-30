import argparse
import sys
from dataclasses import fields, is_dataclass
from typing import Type, Any
import config as cfg_module
from train import train

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
        parser.add_argument(arg_name, type=field_type if field_type in [int, float, str, bool] else str, 
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
        
        # If the target is a list and we got a list, just set it
        if isinstance(current_val, list) and isinstance(value, list):
            setattr(target, field_name, value)
        # If the target is a dict and we got a JSON string, parse it
        elif isinstance(current_val, dict) and isinstance(value, str):
             import json
             try:
                 setattr(target, field_name, json.loads(value))
             except:
                 # If not json, maybe it's a string like "key=val" (simple parsing)
                 pass
        else:
            # Simple type casting
            target_type = type(current_val)
            try:
                if target_type is bool:
                    # Handle boolean flags correctly
                    val = str(value).lower() in ('true', '1', 't')
                    setattr(target, field_name, val)
                else:
                    setattr(target, field_name, target_type(value))
            except ValueError:
                print(f"Warning: Could not cast {key}={value} to {target_type}")

def get_args_parser():
    parser = argparse.ArgumentParser(description="Universal Training CLI")
    add_arguments_from_dataclass(parser, cfg_module.ExperimentConfig)
    return parser

# Usage in train.py
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    
    # 1. Start with default config
    config = cfg_module.ExperimentConfig()
    
    # 2. Apply overrides
    apply_args_to_config(args, config)
    
    # 3. Run
    train(config)
