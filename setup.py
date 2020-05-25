import numpy
import os, sys 
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate=True


setup(
    name='PyRossTSI',
    version='1.0.0',
    url='https://github.com/rajeshrinet/pyrossTSI',
    author='The PyRossTSI team',
    license='MIT',
    description='python library for numerical simulation of infectious disease',
    long_description='PyRossTSI is a library for numerical simulation of infectious disease',
    platforms='works on all platforms (such as LINUX, macOS, and Microsoft Windows)',
    ext_modules=cythonize([ Extension("pyrosstsi/*", ["pyrosstsi/*.pyx"],
        include_dirs=[numpy.get_include()],
        )],
        compiler_directives={"language_level": sys.version_info[0]},
        ),
    libraries=[],
    packages=['pyrosstsi'],
    package_data={'pyrosstsi': ['*.pxd']},
)
