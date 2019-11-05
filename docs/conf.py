# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Try to fix missing autodoc on RTD
# https://stackoverflow.com/a/41078541
import mock

MOCK_MODULES = ['numpy', 'pandas', 'statsmodels', 'matplotlib', 'matplotlib.pyplot',
                'seaborn', 'theano', 'theano.tensor']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- Project information -----------------------------------------------------

project = 'glambox'
copyright = '2019, Felix Molter, Armin W. Thomas'
author = 'Felix Molter, Armin W. Thomas'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'recommonmark',  # support for Markdown files
    'sphinx_markdown_tables',  # support for Markdown formatted tables
    'nbsphinx',  # support for Jupyter Notebooks
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # support for NumPy and Google docstrings
    'sphinx.ext.mathjax'
]

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# https://stackoverflow.com/a/23686917
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
