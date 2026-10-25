"""judgeflake: is your LLM judge flaky?"""

from .cases import Case, load_cases
from .judges import Verdict

__version__ = "0.1.0"
__all__ = ["Case", "Verdict", "load_cases", "__version__"]
