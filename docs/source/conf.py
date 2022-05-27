"""The configuration for sphinx to generate the documentation."""
import datetime
import io
import os
import sys
from typing import List

# the path of the documentation configuration
doc_path = os.path.abspath(os.path.dirname(__file__))

# get the path to the root directory of this project
root_path = os.path.abspath(os.path.join(doc_path, "..", ".."))
sys.path.insert(0, root_path)

# set the base url
html_baseurl = "https://thomasweise.github.io/moptipy/"

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
        else:  # fix all internal urls
            line = line.replace("https://github.com/thomasWeise/"
                                "moptipy/blob/main/LICENSE",
                                "./LICENSE.html")
            for k in [html_baseurl, f"http{html_baseurl[5:]}"]:
                line = line.replace(f"]({k}", "](./")\
                    .replace(f' src="{k}', ' src="./')\
                    .replace(f' href="{k}', ' href="./')

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

# tell sphinx to go kaboom on errors
nitpicky = True
myst_all_links_external = True

# The full version, including alpha/beta/rc tags.
release = {}
with open(os.path.abspath(os.path.sep.join([
        root_path, "moptipy", "version.py"]))) as fp:
    exec(fp.read(), release)  # nosec # nosemgrep
release = release["__version__"]

# The Sphinx extension modules that we use.
extensions = ['myst_parser',  # for processing README.md
              'sphinx.ext.autodoc',  # to convert docstrings to documentation
              'sphinx.ext.doctest',  # do the doc tests again
              'sphinx.ext.intersphinx',  # to link to numpy et al.
              'sphinx_autodoc_typehints',  # to infer types from hints
              'sphinx.ext.viewcode',  # add rendered source code
              ]

# Location of dependency documentation for cross-referencing.
intersphinx_mapping = {
    'abc': ("https://docs.python.org/3/", None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None)
}

# add default values after comma
typehints_defaults = "comma"

# the sources to be processed
source_suffix = ['.rst', '.md']

# Additional files to include.
html_static_path = ["_static"]

# The theme to use for HTML and HTML Help pages.
html_theme = 'bizstyle'

# Code syntax highlighting style:
pygments_style = 'default'
