"""The setup and installation script."""
from setuptools import find_packages, setup

version = {}
with open("moptipy/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='moptipy',
    description='A package for metaheuristic optimization in python.',
    url='git@github.com/thomasWeise/moptipy.git',
    author='Thomas Weise',
    author_email='tweise@ustc.edu.cn',
    version=version["__version__"],
    license='GPL 3.0',
    packages=find_packages(include=['moptipy', 'moptipy.*']),
    package_data={"moptipy.examples.jssp": ["*.txt"]},
    long_description="\n".join([line.strip() for line in
                                open("README.md", "rt").readlines()]),
    long_description_content_type="text/markdown",
    install_requires=[line.strip() for line in
                      open("requirements.txt", "rt").readlines()],
    include_package_data=True,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=[
        "metaheuristics",
        "optimization",
        "hill climber",
        "job shop scheduling"
    ]
)
