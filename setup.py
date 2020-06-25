import numpy
from setuptools import setup, Extension
import setuptools
import os, sys, os.path


setup(
    name='pyrosstsi',
    version='1.0.1',
    url='https://github.com/rajeshrinet/pyrosstsi',
    author='The PyRossTSI team',
    license='MIT',
    description='python library for numerical simulation of infectious disease',
    long_description='pyrosstsi is a library for numerical simulation of infectious disease',
    platforms='works on all platforms (such as LINUX, macOS, and Microsoft Windows)',
    libraries=[],
    packages=['pyrosstsi'],
    install_requires=["cython","numpy","scipy","nlopt"],
)
