#define PY_SSIZE_T_CLEAN
#include <Python.h>             // Python C API
#include <complex>              // For std::complex
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h" // NumPy C API
#include "Faddeeva.hh"

// =============================================================================
// Helper macro to define a Python function that dispatches to real/complex
// C++ backends based on the input array's dtype.
// =============================================================================
// EDITED: Removed the unused 'docstring' parameter to avoid confusion.
#define FADDEEVA_DISPATCH_WRAPPER(py_name, c_name) \
static PyObject* py_name(PyObject* self, PyObject* args) { \
    PyObject* input_obj = NULL; \
    double relerr = 0.0; \
    if (!PyArg_ParseTuple(args, "O|d", &input_obj, &relerr)) { \
        return NULL; \
    } \
\
    PyArray_Descr* descr = PyArray_DescrFromObject(input_obj, NULL); \
    if (descr == NULL) { \
        return NULL; \
    } \
\
    PyObject* result = NULL; \
    if (PyDataType_ISCOMPLEX(descr)) { \
        PyArrayObject* in_array = (PyArrayObject*)PyArray_FROM_OTF( \
            input_obj, NPY_COMPLEX128, 0, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_C_CONTIGUOUS \
        ); \
        if (in_array == NULL) { \
            Py_DECREF(descr); \
            return NULL; \
        } \
        PyArrayObject* out_array = (PyArrayObject*)PyArray_SimpleNew( \
            PyArray_NDIM(in_array), PyArray_DIMS(in_array), NPY_COMPLEX128 \
        ); \
        if (out_array == NULL) { \
            Py_DECREF(in_array); \
            Py_DECREF(descr); \
            return NULL; \
        } \
        std::complex<double>* in_ptr = (std::complex<double>*)PyArray_DATA(in_array); \
        std::complex<double>* out_ptr = (std::complex<double>*)PyArray_DATA(out_array); \
        npy_intp n = PyArray_SIZE(in_array); \
        for (npy_intp i = 0; i < n; ++i) { \
            out_ptr[i] = Faddeeva::c_name(in_ptr[i], relerr); \
        } \
        Py_DECREF(in_array); \
        result = (PyObject*)out_array; \
    } else { \
        PyArrayObject* in_array = (PyArrayObject*)PyArray_FROM_OTF( \
            input_obj, NPY_DOUBLE, 0, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_C_CONTIGUOUS \
        ); \
        if (in_array == NULL) { \
            Py_DECREF(descr); \
            return NULL; \
        } \
        PyArrayObject* out_array = (PyArrayObject*)PyArray_SimpleNew( \
            PyArray_NDIM(in_array), PyArray_DIMS(in_array), NPY_DOUBLE \
        ); \
        if (out_array == NULL) { \
            Py_DECREF(in_array); \
            Py_DECREF(descr); \
            return NULL; \
        } \
        double* in_ptr = (double*)PyArray_DATA(in_array); \
        double* out_ptr = (double*)PyArray_DATA(out_array); \
        npy_intp n = PyArray_SIZE(in_array); \
        for (npy_intp i = 0; i < n; ++i) { \
            out_ptr[i] = Faddeeva::c_name(in_ptr[i]); \
        } \
        Py_DECREF(in_array); \
        result = (PyObject*)out_array; \
    } \
\
    Py_DECREF(descr); \
    return result; \
}

// =============================================================================
// Implement the wrappers for all dispatchable functions using the macro
// =============================================================================
// EDITED: Removed the third 'docstring' argument from macro calls.
FADDEEVA_DISPATCH_WRAPPER(py_erf, erf)
FADDEEVA_DISPATCH_WRAPPER(py_erfc, erfc)
FADDEEVA_DISPATCH_WRAPPER(py_erfi, erfi)
FADDEEVA_DISPATCH_WRAPPER(py_erfcx, erfcx)
FADDEEVA_DISPATCH_WRAPPER(py_Dawson, Dawson)


// =============================================================================
// Wrapper for Faddeeva::w (always complex)
// =============================================================================
static PyObject* py_w(PyObject* self, PyObject* args) {
    PyObject* input_obj = NULL;
    double relerr = 0.0;
    if (!PyArg_ParseTuple(args, "O|d", &input_obj, &relerr)) {
        return NULL;
    }
    
    PyArrayObject* in_array = (PyArrayObject*)PyArray_FROM_OTF(
        input_obj, NPY_COMPLEX128, 0, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_C_CONTIGUOUS
    );
    if (in_array == NULL) {
        return NULL;
    }

    PyArrayObject* out_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(in_array), PyArray_DIMS(in_array), NPY_COMPLEX128
    );
    if (out_array == NULL) {
        Py_DECREF(in_array);
        return NULL;
    }

    std::complex<double>* in_ptr = (std::complex<double>*)PyArray_DATA(in_array);
    std::complex<double>* out_ptr = (std::complex<double>*)PyArray_DATA(out_array);
    npy_intp n = PyArray_SIZE(in_array);

    for (npy_intp i = 0; i < n; ++i) {
        out_ptr[i] = Faddeeva::w(in_ptr[i], relerr);
    }

    Py_DECREF(in_array);
    return (PyObject*)out_array;
}

// =============================================================================
// Wrapper for Faddeeva::w_im (real input, real output)
// =============================================================================
static PyObject* py_w_im(PyObject* self, PyObject* args) {
    PyObject* input_obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &input_obj)) {
        return NULL;
    }
    
    PyArrayObject* in_array = (PyArrayObject*)PyArray_FROM_OTF(
        input_obj, NPY_DOUBLE, 0, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_C_CONTIGUOUS
    );
    if (in_array == NULL) {
        return NULL;
    }

    PyArrayObject* out_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(in_array), PyArray_DIMS(in_array), NPY_DOUBLE
    );
    if (out_array == NULL) {
        Py_DECREF(in_array);
        return NULL;
    }

    double* in_ptr = (double*)PyArray_DATA(in_array);
    double* out_ptr = (double*)PyArray_DATA(out_array);
    npy_intp n = PyArray_SIZE(in_array);

    for (npy_intp i = 0; i < n; ++i) {
        out_ptr[i] = Faddeeva::w_im(in_ptr[i]);
    }

    Py_DECREF(in_array);
    return (PyObject*)out_array;
}


// =============================================================================
// Method and Module Definitions
// =============================================================================

// EDITED: Replaced undeclared variables with the correct string literals.
static PyMethodDef FaddeevaMethods[] = {
    {"w", py_w, METH_VARARGS, "Calculate the Faddeeva function, w(z)."},
    {"w_im", py_w_im, METH_VARARGS, "Calculate Im[w(x)] for real x."},
    {"erf", py_erf, METH_VARARGS, "Calculate the error function, erf(z)."},
    {"erfc", py_erfc, METH_VARARGS, "Calculate the complementary error function, erfc(z)."},
    {"erfi", py_erfi, METH_VARARGS, "Calculate the imaginary error function, erfi(z)."},
    {"erfcx", py_erfcx, METH_VARARGS, "Calculate the scaled complementary error function, erfcx(z)."},
    {"Dawson", py_Dawson, METH_VARARGS, "Calculate the Dawson function, Dawson(z)."},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef faddeeva_module = {
    PyModuleDef_HEAD_INIT,
    "_faddeeva",
    "A C++ extension for the complete Faddeeva function package.",
    -1,
    FaddeevaMethods
};

PyMODINIT_FUNC PyInit__faddeeva(void) {
    PyObject* m = PyModule_Create(&faddeeva_module);
    if (m == NULL) {
        return NULL;
    }
    import_array();
    return m;
}