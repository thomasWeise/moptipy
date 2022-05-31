"""The setup and installation script."""

from setuptools import find_packages, setup

version = {}
with open("moptipy/version.py") as fp:
    exec(fp.read(), version)  # nosec # nosemgrep

setup(
    name='moptipy',
    python_requires='>=3.9',
    description='A package for metaheuristic optimization in python.',
    url='https://thomasweise.github.io/moptipy',
    author='Thomas Weise',
    author_email='tweise@ustc.edu.cn',
    version=version["__version__"],
    license='GPL 3.0',
    packages=find_packages(include=['moptipy', 'moptipy.*']),
    package_data={"moptipy.examples.jssp": ["*.txt"]},
    include_package_data=True,
    long_description="\n".join([line.strip() for line in
                                open("README.md", "rt").readlines()]),
    long_description_content_type="text/markdown",
    install_requires=[line.strip() for line in
                      open("requirements.txt", "rt").readlines()],
    project_urls={
        "Bug Tracker": "https://github.com/thomasWeise/moptipy/issues",
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Framework :: Matplotlib',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Natural Language :: German',
        'Natural Language :: Chinese (Simplified)',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=[
        "metaheuristics",
        "optimization",
        "operations research",
        "evolutionary algorithm",
        "hill climber",
        "experiments",
        "job shop scheduling"
    ]
)
