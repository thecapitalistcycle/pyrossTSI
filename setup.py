import numpy
import setuptools
import os, sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate=True


if 'darwin' == (sys.platform).lower():
    extension = Extension("pyrosstsi/*", ["pyrosstsi/*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-mmacosx-version-min=10.9'],
        extra_link_args=['-mmacosx-version-min=10.9'],
    )
else:
    extension = Extension("pyrosstsi/*", ["pyrosstsi/*.pyx"],
        include_dirs=[numpy.get_include()],
    )


setup(
    name='pyrosstsi',
    version='1.0.1',
    url='https://github.com/rajeshrinet/pyrosstsi',
    author='The PyRossTSI team',
    license='MIT',
    description='python library for numerical simulation of infectious disease',
    long_description='pyrosstsi is a library for numerical simulation of infectious disease',
    platforms='works on all platforms (such as LINUX, macOS, and Microsoft Windows)',
    ext_modules=cythonize([ extension ],
        compiler_directives={"language_level": sys.version_info[0]},
        ),
    libraries=[],
    packages=['pyrosstsi'],
    install_requires=["cython","numpy","scipy","nlopt"],
    package_data={'pyrosstsi': ['*.pxd']},
)
