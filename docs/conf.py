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


# -- Project information -----------------------------------------------------

project = 'CBX'
copyright = '2024, Tim Roith'
author = 'Tim Roith'

# The full version, including alpha/beta/rc tags
release = 'v0.1.6'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autosummary'
]
autosummary_generate = True
#numpydoc_show_class_members = False
numpydoc_show_class_members = False
autodoc_member_order = 'bysource'
autodoc_typehints = "none"
#autoapi_dirs = ['../polarcbo']
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
#autoapi_generate_api_docs = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

nbsphinx_execute = 'never'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'sphinx_rtd_theme'
html_theme = "pydata_sphinx_theme"
html_favicon = '_static/cbx-logo.ico'
#html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "_static/cbx-logo.png"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/PdIPS/CBXpy",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
   "favicons": [
      {
         "rel": "icon",
         "sizes": "16x16",
         "href": "cbx_py32x32.ico",
      },
      {
         "rel": "icon",
         "sizes": "32x32",
         "href": "cbx_py32x32.ico",
      },
   ]
}

nbsphinx_thumbnails = {
    'examples/nns/mnist': '_static/cbx-logo.png',
    'examples/simple_example': '_static/cbx-logo.png',
    'examples/custom_noise': '_static/cbx-logo.png',
    'examples/low_level': '_static/cbx-logo.png',
    'examples/sampling': '_static/cbx-logo.png',
    'examples/success_evaluation': '_static/cbx-logo.png',
}

def setup(app):
    app.add_css_file('css/style.css')