import os
import sys

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'explainability-challenges'
copyright = '2023 Anna Cai, Abhishek Vijayakumar, Jacob Rast'
author = 'Anna Cai, Abhishek Vijayakumar, Jacob Rast'

release = '0.0'
version = '0.0.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Point to package root

sys.path.append(os.path.abspath('../../src/explainability'))
