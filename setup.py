import setuptools
from os import path
import sys

from io import open

here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, 'easylink'))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easylink",
    version="0.0.1",
    author="Fstyle",
    author_email="ofanshen@gmail.com",
    description="Link Prediction models for Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fs302/EasyLink",
    project_urls={
        "Bug Tracker": "https://github.com/fs302/EasyLink/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "easylink"},
    packages=setuptools.find_packages(where="easylink"),
    python_requires=">=3.6",
)