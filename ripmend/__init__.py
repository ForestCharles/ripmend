"""Repair damaged CD rips using other rips of the same disc."""

try:
    from ._version import version as __version__
except ImportError:  # running from a source tree that was never installed
    __version__ = "unknown"
