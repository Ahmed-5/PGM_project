"""Command-line entry point for the REMUL dynamics experiments.

Usage:
    python -m remul.cli --data.name md17 --data.molecule aspirin \
        --model.name egnn --train.mode remul --train.penalty constant \
        --train.beta 1.0 --train.epochs 1 --train.max_steps 5

Any dotted ``--section.field value`` overrides the corresponding config field
(section in {data, model, train, log}). Values are parsed with ``ast.literal_eval``
when possible, otherwise kept as strings; booleans and numbers are coerced to the
field's existing type.
"""
from __future__ import annotations

import ast
import sys

from .config import ExperimentConfig
from .train import train


def _coerce(value: str, current):
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    if current is None:
        return parsed
    if isinstance(current, bool):
        if isinstance(parsed, str):
            return parsed.lower() in ("1", "true", "yes")
        return bool(parsed)
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(parsed)
        except (ValueError, TypeError):
            return parsed
    if isinstance(current, float):
        try:
            return float(parsed)
        except (ValueError, TypeError):
            return parsed
    return parsed


def parse_args(argv) -> ExperimentConfig:
    cfg = ExperimentConfig()
    sections = {"data": cfg.data, "model": cfg.model, "train": cfg.train, "log": cfg.log}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected argument: {tok}")
        key = tok[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            i += 1
        else:
            value = argv[i + 1]
            i += 2
        if "." not in key:
            raise ValueError(f"Override must be --section.field, got --{key}")
        section, field = key.split(".", 1)
        if section not in sections:
            raise ValueError(f"Unknown section '{section}'. Options: {sorted(sections)}")
        obj = sections[section]
        if not hasattr(obj, field):
            raise ValueError(f"Unknown field '{field}' in section '{section}'")
        setattr(obj, field, _coerce(value, getattr(obj, field)))
    return cfg


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = parse_args(argv)
    train(cfg)


if __name__ == "__main__":
    main()
