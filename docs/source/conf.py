"""The configuration for sphinx to generate the documentation."""
from typing import Final

from pycommons.dev.doc.setup_doc import setup_doc
from pycommons.io.path import Path, file_path

# the path of the documentation configuration
doc_path: Final[Path] = file_path(__file__).up(1)
root_path: Final[Path] = doc_path.up(2)
setup_doc(doc_path, root_path, 2023, dependencies=(
    "matplotlib", "numpy", "pycommons", "scipy", "sklearn"),
    full_urls={
        "https://github.com/thomasWeise/moptipy/blob/main/LICENSE":
            "./LICENSE.html"},
    static_paths=("_static", ))
