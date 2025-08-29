from setuptools import setup, Extension
import numpy

# Define the C++ extension module
faddeeva_module = Extension(
    'intelligen.special._faddeeva',  # The full name of the module once installed
    sources=['src/Faddeeva.cc', 'src/_faddeeva_wrapper.cpp'],
    include_dirs=[numpy.get_include()],  # Add NumPy headers
    language='c++',
)

setup(
    name='intelligen',
    version='0.13.1',
    packages=['intelligen', 'intelligen.special'],
    ext_modules=[faddeeva_module],
    install_requires=['numpy'],
)
