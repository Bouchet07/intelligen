project('intelligen', 'cpp',
  version: '0.13.1',
  default_options: ['cpp_std=c++11'],
)

py_mod = import('python')
# pure: false is essential because this package contains a compiled extension
py3 = py_mod.find_installation('python3', pure: false)

# 1. Get NumPy include directories
incdir_numpy = run_command(py3,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

# 2. Install the Pure Python package
# This copies everything inside 'intelligen/' to 'site-packages/intelligen/'
install_subdir('intelligen', install_dir: py3.get_install_dir())

# 3. Build and Install the C++ Extension
py3.extension_module(
  '_faddeeva',  # The name of the resulting .pyd file
  ['src/_faddeeva_wrapper.cpp', 'src/Faddeeva.cc'],
  include_directories: [include_directories('src'), include_directories(incdir_numpy)],
  subdir: 'intelligen/special', # Installs into 'intelligen/special/'
  dependencies: [py3.dependency()],
  install: true
)