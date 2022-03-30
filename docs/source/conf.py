"""The configuration for sphinx to generate the documentation."""
import datetime
import io
import os
import sys
from typing import List


# the path of the documentation
doc_path = os.path.abspath(os.path.dirname(__file__))

# get the path to the root directory of this project
root_path = os.path.abspath(os.path.join(doc_path, "..", ".."))
sys.path.insert(0, root_path)

# We want to include the contents of our GitHub README.md file.
# So first, we need to load the README.md file.
old_lines: List[str]
with io.open(os.path.join(root_path, "README.md"),
             "rt", encoding='utf-8-sig') as reader:
    old_lines = reader.readlines()

# Now, we need to fix the file contents.
# We discard the top-level heading as well as the badge for the build status.
# We need to move all sub-headings one step up.
new_lines = []
in_code: bool = False  # we only process non-code lines
skip: bool = True
for line in old_lines:
    if skip:  # we skip everything until the introduction section
        if line.lstrip().startswith("## 1. Introduction"):
            skip = False
        else:
            continue
    if in_code:
        if line.startswith("```"):
            in_code = False  # toggle to non-code
    else:
        if line.startswith("```"):
            in_code = True  # toggle to code
        elif line.startswith("#"):
            line = line[1:]  # move all sub-headings one step up
    new_lines.append(line)

with io.open(os.path.join(doc_path, "README.md"), "wt",
             encoding="utf-8") as outf:
    outf.writelines(new_lines)

# enable myst header anchors
myst_heading_anchors = 6

# project information
project = 'moptipy'
author = 'Thomas Weise'
# noinspection PyShadowingBuiltins
copyright = f"2021-{datetime.datetime.now ().year}, {author}"

# The full version, including alpha/beta/rc tags
release = {}
with open(os.path.abspath(os.path.sep.join([
        root_path, "moptipy", "version.py"]))) as fp:
    exec(fp.read(), release)
release = release["__version__"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'myst_parser']

# the sources to be processed
source_suffix = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bizstyle'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
