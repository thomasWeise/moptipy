"""The setup and installation script."""

from re import compile as re_compile
from re import sub as re_sub
from typing import Final, Pattern

from setuptools import setup

# We want to use our README.md as project description.
# However, we must fix all the references inside.
with open("README.md", encoding="utf-8-sig") as reader:
    old_lines = reader.readlines()

# It seems that the markdown parser does not auto-generate anchors. This means
# that we need to fix all references following the pattern `[xxx](#12-hello)`
# to `[xxx]({docu_url#hello)`, where `docu_url` is the url of our
# documentation. We do this with a regular expression `regex_search`.
new_lines: Final[list[str]] = []
in_code: bool = False  # we only process non-code lines

# the base url where the documentation will land
doc_url: Final[str] = "https://thomasweise.github.io/moptipy"
# the url of our repository
repo_url: Final[str] = "https://github.com/thomasWeise/moptipy"

# detects strings of the form [xyz](#123-bla) and gives \1=xyz and \2=bla
regex_search: Final[Pattern] = re_compile("(\\[.+?])\\(#\\d+-(.+?)\\)")
regex_repl: Final[str] = f"\\1({doc_url}#\\2)"

# other replacements
license_old: Final[str] = f"{repo_url}/blob/main/LICENSE"
license_new: Final[str] = f"{doc_url}/LICENSE.html"

for full_line in old_lines:
    line: str = str.rstrip(full_line)
    if in_code:
        if line.startswith("```"):
            in_code = False  # toggle to non-code
    elif line.startswith("```"):
        in_code = True  # toggle to code
    else:  # fix all internal urls
        # replace links of the form "#12-bla" to "#bla"
        line = re_sub(regex_search, regex_repl, line)
        line = str.replace(line, license_old, license_new)
    new_lines.append(line)

# Now we can use the code in the setup.
setup(long_description="\n".join(new_lines),
      long_description_content_type="text/markdown")
