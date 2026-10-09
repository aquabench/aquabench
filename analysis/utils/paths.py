"""Shared results-directory resolution, mirroring the $DATA_ROOT convention.

$RESULTS_DIR selects which results tree to read/write (default "results") -- e.g. set
to results_archive_pre_bgfix or results_dinovitb_bgfix_pilot to keep an old or a pilot
result set fully separate from the current default. Within a results tree, the metric
used for ranking (test_miou vs test_miou_all) further splits the *analysis output* dir
so a --metric=miou run and a --metric=miou_all run never overwrite each other.
"""

from __future__ import annotations

import os
from pathlib import Path


def analysis_dir(metric: str = "miou") -> Path:
    root = Path(os.environ.get("RESULTS_DIR", "results"))
    return root / ("analysis" if metric == "miou" else "analysis_miou_all")
