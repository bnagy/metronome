import setuptools
from distutils.core import setup
import os

setup(
    name="Metronome",
    version="0.0.1",
    author="Ben Nagy",
    packages=["metronome"],
    license="3-Clause BSD",
    url="https://github.com/bnagy/metronome",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    install_requires=[
        "setuptools",
        "pandas",
        "Bio",
    ],
)
