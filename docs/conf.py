import sphinx_rtd_theme
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from packaging.version import parse
from small_text.version import __version__ as version

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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'small-text'
copyright = '2020–2023, Christopher Schröder'
author = 'Christopher Schröder'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'm2r2',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md']

# Suppress warning caused by duplicate labels from the autosectionlabel extension.
suppress_warnings = ['autosectionlabel.*']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 2
}

# -- sphinx.ext.autodoc------ -------------------------------------------------

autodoc_member_order = 'bysource'

autodoc_preserve_defaults = True

# -- sphinx.ext.autosectionlabel------ ----------------------------------------

autosectionlabel_prefix_document = True

# -- Document variables------ -------------------------------------------------

rst_prolog = """
.. |LIBRARY_VERSION| replace:: """ + version + """
.. |CUDA_VERSION| replace:: 10.1
"""

# -- Layout------------------- ------------------------------------------------

html_context = {
    'github_url': 'https://github.com/webis-de/small-text'
}

html_css_files = [
    'css/custom.css',
]

# -- Linkcode resolve------------------- --------------------------------------

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    # only link to classes
    if '.' in info['fullname']:
        return None
    filename = info['module'].replace('.', '/') + '.py'
    return 'https://github.com/webis-de/small-text/blob/v%s/%s' % (str(parse(version)), filename)

# -- Intersphinx------------------- -------------------------------------------

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/master/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None)
}
