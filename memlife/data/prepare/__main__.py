"""``python -m memlife.data.prepare <step>`` -- one entry point for the whole preparation.

Each step is a module with its own ``main()``; this only routes, so ``--help`` on a step is
that step's real help and no flag is duplicated here.
"""
from __future__ import annotations

import importlib
import sys

STEPS = ("transcode", "supermemory", "egolife")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        print(f"usage: python -m memlife.data.prepare <{'|'.join(STEPS)}> [step args]",
              file=sys.stderr)
        return 2
    step = sys.argv.pop(1)
    sys.argv[0] = f"memlife.data.prepare {step}"
    # Imported, not run as __main__: the steps hand module-level workers to a process pool,
    # which must be able to find them by module name.
    return importlib.import_module(f"memlife.data.prepare.{step}").main()


if __name__ == "__main__":
    raise SystemExit(main())
