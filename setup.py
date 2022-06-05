"""The setup and installation script."""

import io
import re

from setuptools import setup

# We want to use our README.md as project description.
# However, we must fix all the references inside.
with io.open("README.md",
             "rt", encoding='utf-8-sig') as reader:
    old_lines = reader.readlines()

# It seems that the markdown parser does not auto-generate anchors. This means
# that we need to fix all references following the pattern `[xxx](#12-hello)`
# to `[xxx]({docu_url#hello)`, where `docu_url` is the url of our
# documentation. We do this with a regular expression `regex_search`.
new_lines = []
in_code: bool = False  # we only process non-code lines
# detects strings of the form [xyz](#123-bla) and gives \1=xyz and \2=bla
regex_search = re.compile("(\\[.+?])\\(#\\d+-(.+?)\\)")
regex_repl: str = "\\1(https://thomasweise.github.io/moptipy/index.html#\\2)"
license_old: str = "https://github.com/thomasWeise/moptipy/blob/main/LICENSE"
license_new: str = "https://thomasweise.github.io/moptipy/LICENSE.html"

for line in old_lines:
    line = line.rstrip()
    if in_code:
        if line.startswith("```"):
            in_code = False  # toggle to non-code
    else:
        if line.startswith("```"):
            in_code = True  # toggle to code
        else:  # fix all internal urls
            # replace links of the form "#12-bla" to "#bla"
            line = re.sub(regex_search, regex_repl, line)
            line = line.replace(license_old, license_new)
    new_lines.append(line)

# Now we can use the code in the setup.
setup(long_description="\n".join(new_lines),
      long_description_content_type="text/markdown")
