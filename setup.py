import io
import os
import re
from os import path

from setuptools import find_packages
from setuptools import setup


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="optim-rl",
    version="0.1.0",
    url="https://github.com/shyamsn97/ez-rl",
    license='MIT',

    author="Shyam Sudhakaran",
    author_email="shyamsnair@protonmail.com",

    description="Easy, flexible, and modular RL components designed as torch optimizers.",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'torch',
        'gym',
        'scipy',
        'einops',
        'tqdm'
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
