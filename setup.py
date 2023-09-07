from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

board_module = Pybind11Extension(
    'board',
    ['src/board.cpp', 'src/bindings.cpp'],
    include_dirs=['include'],
    extra_compile_args=['-O3', '-DASIO_STANDALONE']
)

setup(
    name='board',
    version='0.1',
    author='Aniruddha Deb',
    author_email='aniruddha.deb.2002@gmail.com',
    description='Barebones pybind11+setup.py example project',
    ext_modules=[board_module],
    cmdclass={"build_ext": build_ext},
)
