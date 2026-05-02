cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# --- Faddeeva Extension ---
nanobind_add_module(
    _faddeeva
    src/_faddeeva_wrapper.cpp
    src/Faddeeva.cc
)
target_include_directories(_faddeeva PRIVATE src)

# --- erfinv Extension ---
nanobind_add_module(
    _erfinv
    src/_erfinv_wrapper.cpp
)
target_include_directories(_erfinv PRIVATE src)

# --- Installation ---
install(TARGETS _faddeeva _erfinv LIBRARY DESTINATION intelligen/special)
