"""FastExcerpt namespace."""

from importlib_metadata import PackageNotFoundError, version

from .fastexcerpt import FastExcerpt, SubwordFastExcerpt

__author__ = "Kevin Alex Zhang"
__email__ = "hello@kevz.dev"

# Used to automatically set version number from github actions
# as well as not break when being tested locally
try:
    __version__ = version(__package__)  # type: ignore
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


__all__ = [
    "__version__",
    "FastExcerpt",
    "SubwordFastExcerpt",
]
