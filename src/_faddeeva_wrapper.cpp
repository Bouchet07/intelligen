// src/_faddeeva_wrapper.cpp

#define PY_SSIZE_T_CLEAN
#include <Python.h>           // Python C API
#include <complex>            // For std::complex
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h" // NumPy C API
#include "Faddeeva.hh"         // Your Faddeeva header

// 1. Wrapper for a REAL function: Faddeeva::erf(double)
static PyObject* py_erf(PyObject* self, PyObject* args) {
    PyArrayObject* in_array;

    // Parse the input from Python: "O!" checks if the object is a PyArray_Type
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)) {
        return NULL; // Error
    }

    // Ensure the input array is of type double (NPY_DOUBLE) and is C-style contiguous
    in_array = (PyArrayObject*)PyArray_ContiguousFromObject((PyObject*)in_array, NPY_DOUBLE, 1, 1);
    if (in_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Input must be a 1D numpy array of floats.");
        return NULL;
    }

    // Create the output NumPy array with the same shape as the input
    PyArrayObject* out_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(in_array), PyArray_DIMS(in_array), NPY_DOUBLE
    );
    if (out_array == NULL) {
        Py_DECREF(in_array);
        return NULL;
    }

    // Get pointers to the data buffers
    double* in_ptr = (double*)PyArray_DATA(in_array);
    double* out_ptr = (double*)PyArray_DATA(out_array);
    npy_intp n = PyArray_SIZE(in_array);

    // The core loop: call the C++ function for each element
    for (npy_intp i = 0; i < n; ++i) {
        out_ptr[i] = Faddeeva::erf(in_ptr[i]);
    }

    Py_DECREF(in_array); // Clean up the input array reference
    return (PyObject*)out_array; // Return the new output array
}


// 2. Wrapper for a COMPLEX function: Faddeeva::w(complex)
static PyObject* py_w(PyObject* self, PyObject* args) {
    PyArrayObject* in_array;
    double relerr = 0.0; // Default relative error

    // "O!|d": Array object OR double for optional relerr
    if (!PyArg_ParseTuple(args, "O!|d", &PyArray_Type, &in_array, &relerr)) {
        return NULL;
    }
    
    // Ensure input is complex128 and contiguous
    in_array = (PyArrayObject*)PyArray_ContiguousFromObject((PyObject*)in_array, NPY_COMPLEX128, 1, 1);
    if (in_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Input must be a 1D numpy array of complex numbers.");
        return NULL;
    }

    // Create output array
    PyArrayObject* out_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(in_array), PyArray_DIMS(in_array), NPY_COMPLEX128
    );
    if (out_array == NULL) {
        Py_DECREF(in_array);
        return NULL;
    }

    // Get data pointers
    std::complex<double>* in_ptr = (std::complex<double>*)PyArray_DATA(in_array);
    std::complex<double>* out_ptr = (std::complex<double>*)PyArray_DATA(out_array);
    npy_intp n = PyArray_SIZE(in_array);

    // The core loop
    for (npy_intp i = 0; i < n; ++i) {
        out_ptr[i] = Faddeeva::w(in_ptr[i], relerr);
    }

    Py_DECREF(in_array);
    return (PyObject*)out_array;
}


// 3. Method definition table: maps Python function names to C++ functions
static PyMethodDef FaddeevaMethods[] = {
    {"erf", py_erf, METH_VARARGS, "Calculate the error function for a NumPy array of real numbers."},
    {"w", py_w, METH_VARARGS, "Calculate the Faddeeva function for a NumPy array of complex numbers."},
    {NULL, NULL, 0, NULL} // Sentinel
};

// 4. Module definition structure
static struct PyModuleDef faddeeva_module = {
    PyModuleDef_HEAD_INIT,
    "_faddeeva", // Module name
    "A C++ extension for Faddeeva functions.", // Module docstring
    -1,
    FaddeevaMethods
};

// 5. Module initialization function
PyMODINIT_FUNC PyInit__faddeeva(void) {
    PyObject* m = PyModule_Create(&faddeeva_module);
    if (m == NULL) {
        return NULL;
    }
    // IMPORTANT: This macro MUST be called to initialize the NumPy C API
    import_array();
    return m;
}