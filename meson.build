project('intelligen', 'cpp', 'c',
        version: '0.13.1',
        default_options: ['cpp_std=c++11'])

py = import('python').find_installation(pure: false)

# Get NumPy compile/link flags via Python (recommended)
numpy_dep = dependency('numpy')

py.extension_module(
  'intelligen.special._faddeeva',
  ['src/Faddeeva.cc', 'src/_faddeeva_wrapper.cpp'],
  dependencies: [numpy_dep],
  install: true
)

# Install the entire Python package tree into site-packages
install_subdir('intelligen', install_dir: py.get_install_dir())