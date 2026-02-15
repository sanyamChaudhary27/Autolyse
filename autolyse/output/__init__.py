"""Output handlers for different output formats"""

from .jupyter_display import JupyterDisplay
from .html_generator import HTMLGenerator

__all__ = [
    "JupyterDisplay",
    "HTMLGenerator",
]
