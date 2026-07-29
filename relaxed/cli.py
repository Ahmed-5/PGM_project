"""Unified dotted-flag CLI for the relaxed-equivariance framework.

Any nested config field is settable: ``--section.field value`` (e.g.
``--data.name QM9 --model.name gcn --train.epochs 50``). Lists accept
space-separated tokens or a Python/JSON list literal; dicts accept a
Python/JSON dict literal; booleans accept true/1/t and false/0/f.

The task engine is inferred from the dataset family (see
``relaxed.config.TASK_BY_DATASET``). Multiple seeds run in a loop:
``--run.seeds "0 1 2"``.
"""
from __future__ import annotations

import argparse
import ast
import json
import typing
from dataclasses import fields, is_dataclass
from typing import Any, Type

from .config import ExperimentConfig


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).lower()
    if v in ("true", "1", "t", "yes", "y"):
        return True
    if v in ("false", "0", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'")


def add_arguments_from_dataclass(parser: argparse.ArgumentParser, dtype: Type,
                                 prefix: str = "") -> None:
    # Resolve string annotations (config.py uses `from __future__ import annotations`).
    hints = typing.get_type_hints(dtype)
    for field in fields(dtype):
        field_type = hints[field.name]
        if is_dataclass(field_type):
            add_arguments_from_dataclass(parser, field_type,
                                         prefix=f"{prefix}{field.name}.")
            continue
        arg_name = f"--{prefix}{field.name}"
        if hasattr(field_type, "__origin__"):
            origin = field_type.__origin__
            if origin is list:
                parser.add_argument(arg_name, nargs="+", help=arg_name)
                continue
            if origin is dict:
                parser.add_argument(arg_name, type=str, help=f"JSON/dict for {arg_name}")
                continue
        if field_type is bool:
            parser.add_argument(arg_name, type=str2bool, required=False, help=arg_name)
        else:
            parser.add_argument(arg_name,
                                type=field_type if field_type in (int, float, str) else str,
                                required=False, help=arg_name)


def _parse_literal(value: Any):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                return json.loads(value)
            except Exception:
                return value
    return value


def apply_args_to_config(args: argparse.Namespace, config: ExperimentConfig) -> set:
    """Apply CLI overrides; returns the set of dotted keys actually provided."""
    overridden = set()
    for key, value in vars(args).items():
        if value is None:
            continue
        overridden.add(key)
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            target = getattr(target, part)
        field_name = parts[-1]
        current = getattr(target, field_name)

        if isinstance(current, list):
            parsed = value
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                literal = _parse_literal(value[0])
                if isinstance(literal, (list, tuple)):
                    parsed = list(literal)
                else:
                    # single shell token like "0 1 2" or "so3" -> split/keep
                    parsed = value[0].split() if " " in value[0] else [value[0]]
            elif isinstance(value, str):
                literal = _parse_literal(value)
                parsed = list(literal) if isinstance(literal, (list, tuple)) else [value]
            # coerce element types to match existing entries
            if current and parsed:
                elem_type = type(current[0])
                try:
                    parsed = [elem_type(v) for v in parsed]
                except (ValueError, TypeError):
                    pass
            setattr(target, field_name, parsed)
        elif isinstance(current, dict) and isinstance(value, str):
            literal = _parse_literal(value)
            if isinstance(literal, dict):
                setattr(target, field_name, literal)
            else:
                print(f"Warning: could not parse dict for {key}={value}")
        elif isinstance(current, bool):
            setattr(target, field_name, str2bool(value))
        elif current is None:
            setattr(target, field_name, _parse_literal(value))
        else:
            target_type = type(current)
            try:
                setattr(target, field_name, target_type(value))
            except (ValueError, TypeError):
                print(f"Warning: could not cast {key}={value} to {target_type}")
    return overridden


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified relaxed-equivariance CLI")
    add_arguments_from_dataclass(parser, ExperimentConfig)
    return parser


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build config from CLI args, then finalize with the final values."""
    config = ExperimentConfig.__new__(ExperimentConfig)
    # Bypass __post_init__ on first construction: set defaults per section.
    from .config import (DataConfig, ModelConfig, LossConfig, ScheduleConfig,
                         TrainConfig, LogConfig, RunConfig)
    config.data = DataConfig()
    config.model = ModelConfig()
    config.loss = LossConfig()
    config.schedule = ScheduleConfig()
    config.train = TrainConfig()
    config.log = LogConfig()
    config.run = RunConfig()
    config._overridden = apply_args_to_config(args, config)
    config.finalize()
    return config


def main():
    args = get_args_parser().parse_args()
    config = build_config(args)

    for seed in config.run.seeds:
        config.train.seed = seed
        if config.train.task == "dynamics":
            from .engines.dynamics import train
        else:
            from .engines.graph import train
        train(config)


if __name__ == "__main__":
    main()
