"""Run train_iqp_affine_mlp.py on ONE lensing class.

data_lens_class.install(cls) swaps the class's images into data_lens_r3 before the trainer
imports anything from it, which is how every other per-class arm in this repo is run
(seeds/select_checkpoint.py does the same before scoring).  Usage:

    python3 seeds/run_mlp_class.py <class> [trainer args ...]

The output directory must be named "<class>_..." so evaluate_v11.py's class detection picks
the class up from the directory name as well as from meta.json.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import data_lens_class                                    # noqa: E402


def main(argv):
    if len(argv) < 2:
        raise SystemExit("usage: run_mlp_class.py <class> [trainer args ...]")
    cls = argv[0]
    if cls not in data_lens_class.CLASSES:
        raise SystemExit("unknown class %r, expected one of %s" % (cls, data_lens_class.CLASSES))
    data_lens_class.install(cls)
    print("[class] installed %r (installed_class=%r)"
          % (cls, data_lens_class.installed_class()), flush=True)
    import train_iqp_affine_mlp as T                       # imported AFTER install
    return T.main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
