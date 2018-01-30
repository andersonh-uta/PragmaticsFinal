from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include
from shutil import move

try:
    setup(
        name="ppmi",
        ext_modules=cythonize("cython_src/ppmi.pyx"),
        include_dirs=get_include()
    )
except:
    print("ERROR!")
    print("Could not compile Cython code.  Please consult the Cython documentation "
          "for information on how to configure Cython.  This program will fall back "
          "on a pure Python implementation of PPMI matrix generation, which will be "
          "considerable slower than the Cython version, but will still run.")