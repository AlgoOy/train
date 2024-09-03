import os, sys
import shutil
import datetime


from setuptools import setup, find_packages
from setuptools.command.install import install


readme = open('README.md').read()
VERSION = '0.0.1'

requirements = [
    "numpy",
    "torch",
    "torchvision",
    "PIL",
]

VERSION += "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")

setup(
    name="trainidsnet",
    version=VERSION,
    author="AlgoOy",
    author_email="AlgoOy@stu.ecnu.eud.cn",
    url="https://github.com/AlgoOy/train",
    description="train",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("*test*",)),
    zip_safe=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
