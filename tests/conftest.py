import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent

# Add virtual environment site-packages first so updated dependencies take precedence
for p in sorted(root_dir.glob(".venv/lib/python*/site-packages"), reverse=True):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Add repository root to sys.path so 'src' can be imported everywhere
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
