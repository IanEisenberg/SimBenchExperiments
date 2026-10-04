"""Make the src-layout ``simbench_exp`` package importable in tests without relying on
an editable install. Prepending ``src`` guarantees the working copy's source
(including a git worktree's own ``src``) is used, not a globally installed one.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
