import pkg_resources

from .skies import (
    print_magnitude_overview,
)


try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = []
