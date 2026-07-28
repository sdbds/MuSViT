"""Direct-module adapter for the sole staff-level OMR v2 runtime."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .arguments import config_from_namespace, parser_resume, parser_train
from .protocol.runtime import resume as resume_training
from .protocol.runtime import train as run_training


def train(args: argparse.Namespace) -> str:
    """Run a normalized argparse launch through protocol v2."""
    return str(run_training(config_from_namespace(args)))


def resume(args: argparse.Namespace) -> str:
    """Resume from an argparse namespace containing only allowed overrides."""
    values = vars(args).copy()
    run_dir = values.pop("run_dir")
    return str(resume_training(run_dir, **values))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "resume":
        result = resume(parser_resume.parse_args(arguments[1:]))
    else:
        if arguments and arguments[0] == "train":
            arguments = arguments[1:]
        result = train(parser_train.parse_args(arguments))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
