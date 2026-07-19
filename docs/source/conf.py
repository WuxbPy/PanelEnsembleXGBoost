"""
Sphinx configuration for PanelEnsembleXGBoost documentation.
"""

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# Add the project root to the Python path so autodoc can find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# -- Project information -----------------------------------------------------

project = 'PanelEnsembleXGBoost'
copyright = f'2026, PanelEnsembleXGBoost Research Team'
author = 'PanelEnsembleXGBoost Research Team'

# The full version, including alpha/beta/rc tags
release = '2.1.0'
version = '2.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    # Core Sphinx extensions
    'sphinx.ext.autodoc',        # Auto-generate docs from docstrings
    'sphinx.ext.autosummary',    # Generate summary tables for modules
    'sphinx.ext.napoleon',       # Support NumPy/Google style docstrings
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.intersphinx',    # Link to other projects' documentation
    'sphinx.ext.todo',           # Support todo directives
    'sphinx.ext.coverage',       # Document coverage checks
    'sphinx.ext.mathjax',        # Support LaTeX math rendering
    'sphinx.ext.githubpages',    # Create .nojekyll for GitHub Pages

    # Third-party extensions
    'sphinx_rtd_theme',          # Read the Docs theme
    'sphinx_autodoc_typehints',  # Auto-document type hints
    'numpydoc',                  # NumPy docstring conventions
    'myst_parser',               # Markdown support
]

# -- Autodoc settings --------------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'member-order': 'bysource',
}

autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# -- Napoleon settings (NumPy/Google docstring support) ----------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True

# -- Numpydoc settings ------------------------------------------------------

numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'xgboost': ('https://xgboost.readthedocs.io/en/stable', None),
    'statsmodels': ('https://www.statsmodels.org/stable', None),
}

# -- Templates path ----------------------------------------------------------

templates_path = ['_templates']

# -- Source file suffixes ----------------------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Master document ---------------------------------------------------------

master_doc = 'index'

# -- Language ----------------------------------------------------------------

language = 'en'

# -- Exclude patterns --------------------------------------------------------

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output settings ----------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_static_path = ['_static']

# Custom CSS (if needed)
html_css_files = [
    'css/custom.css',
]

# -- Todo settings -----------------------------------------------------------

todo_include_todos = True

# -- Autosummary settings ----------------------------------------------------

autosummary_generate = True
