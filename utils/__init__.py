"""Analysis and diagnostic scripts for the loop pipeline.

`plc_main.py` stays at the repository root; the pipeline it runs lives in
`model` (`mini_nn`, `loop_extraction`, `warmup_cube`), and everything here
is a follow-up reading of a finished run.  Scripts can be run as
`python utils/<script>.py` or `python -m utils.<script>`.
"""

from . import _paths as _paths