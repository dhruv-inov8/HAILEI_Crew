# tools/__init__.py
"""
HAILEI Educational Tools Package - Modern CrewAI Implementation

This package contains specialized tools for the HAILEI Course Design System using
the modern @tool decorator approach from CrewAI:
- blooms_taxonomy_tool: Learning objective validation and generation
- accessibility_checker_tool: WCAG compliance and UDL validation  
- resource_search_tool: Educational resource discovery and curation
"""

from .blooms_taxonomy_tool import blooms_taxonomy_tool
from .accessibility_checker_tool import accessibility_checker_tool
from .resource_search_tool import resource_search_tool

__all__ = [
    'blooms_taxonomy_tool',
    'accessibility_checker_tool', 
    'resource_search_tool'
]

# Version info
__version__ = "2.0.0"
__author__ = "HAILEI Development Team"
__description__ = "Modern Educational Intelligence Tools for Course Design"