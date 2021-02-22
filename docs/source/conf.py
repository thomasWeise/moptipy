# path setup
import os
import sys

root_path = os.path.abspath(os.path.sep.join([os.path.dirname(__file__), "..", ".."]))
sys.path.insert(0, root_path)

# project information

project = 'moptipy'
author = 'Thomas Weise'
copyright = '2021, ' + author

# The full version, including alpha/beta/rc tags
release = {}
with open(os.path.abspath(os.path.sep.join([root_path, "moptipy", "version.py"]))) as fp:
    exec(fp.read(), release)
release = release["__version__"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc']

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'haiku'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
