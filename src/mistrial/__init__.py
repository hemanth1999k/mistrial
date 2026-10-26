"""mistrial: is your LLM judge flaky?"""

from .cases import Case, load_cases
from .judges import StabilizedJudge, Verdict, stabilize

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("mistrial")  # single source of truth: pyproject.toml
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0+unknown"
__all__ = ["Case", "Verdict", "StabilizedJudge", "stabilize", "load_cases", "__version__"]
