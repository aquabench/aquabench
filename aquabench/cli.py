"""Command line: ``aquabench download`` and ``aquabench verify``.

    aquabench download --data-root /data/aquabench [--datasets SUIM COU] [--source <repo-or-dir>]
    aquabench verify   --data-root /data/aquabench
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aquabench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="download datasets into --data-root")
    dl.add_argument("--data-root", required=True)
    dl.add_argument("--datasets", nargs="+", default=["all"])
    dl.add_argument("--source", default=None, help="Hugging Face dataset repo id or local unified dir")
    dl.add_argument("--force", action="store_true", help="re-download datasets that are already present")

    vf = sub.add_parser("verify", help="check the layout of --data-root")
    vf.add_argument("--data-root", required=True)
    vf.add_argument("--datasets", nargs="+", default=["all"])

    args = parser.parse_args(argv)
    datasets = "all" if args.datasets == ["all"] else args.datasets

    from aquabench.data import verify_data_root
    if args.cmd == "download":
        from aquabench.download import download
        download(args.data_root, datasets, source=args.source, force=args.force)

    problems = verify_data_root(args.data_root, datasets)
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    if not problems:
        print(f"{args.data_root}: OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
