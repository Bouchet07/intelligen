#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <complex>
#include <vector>
#include <execution>
#include "Faddeeva.hh"

// Cross-platform support for alloca
#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

namespace nb = nanobind;
using namespace nb::literals;

// Improved helper function: handles read-only input, drops GIL, optimizes shape allocation
template <typename OutT, typename InT, typename Func>
nb::ndarray<nb::numpy, OutT, nb::c_contig> apply_func(nb::ndarray<const InT, nb::c_contig> input, Func func) {
    size_t ndim = input.ndim();
    
    // Use a stack-allocated buffer for shape since ndim is typically very small
    size_t* shape = (size_t*)alloca(ndim * sizeof(size_t)); 
    for (size_t i = 0; i < ndim; ++i) shape[i] = input.shape(i);
    
    OutT* data = new OutT[input.size()];
    nb::capsule owner(data, [](void *p) noexcept { delete[] (OutT *) p; });
    nb::ndarray<nb::numpy, OutT, nb::c_contig> out_arr(data, ndim, shape, owner);
    
    const InT* in_ptr = input.data();
    OutT* out_ptr = out_arr.data();
    
    // Release the GIL before heavy computation to allow other Python threads to run
    nb::gil_scoped_release release; 
    
    std::transform(std::execution::par_unseq, in_ptr, in_ptr + input.size(), out_ptr, func);
    
    return out_arr;
}

NB_MODULE(_faddeeva, m) {
    // ------------------------------------------------------------------------
    // w
    // ------------------------------------------------------------------------
    m.def("w", [](nb::ndarray<const std::complex<double>, nb::c_contig> input, double relerr) {
        return apply_func<std::complex<double>>(input, [&](std::complex<double> z) { return Faddeeva::w(z, relerr); });
    }, "input"_a, "relerr"_a = 0, "Calculate the Faddeeva function, w(z).");
    m.def("w", [](std::complex<double> z, double relerr) { return Faddeeva::w(z, relerr); }, "z"_a, "relerr"_a = 0);

    // ------------------------------------------------------------------------
    // w_im
    // ------------------------------------------------------------------------
    m.def("w_im", [](nb::ndarray<const double, nb::c_contig> input) {
        return apply_func<double>(input, [&](double x) { return Faddeeva::w_im(x); });
    }, "input"_a, "Calculate Im[w(x)] for real x.");
    m.def("w_im", [](double x) { return Faddeeva::w_im(x); }, "x"_a);

    // ------------------------------------------------------------------------
    // erf
    // ------------------------------------------------------------------------
    m.def("erf", [](nb::ndarray<const double, nb::c_contig> input) {
        return apply_func<double>(input, [&](double x) { return Faddeeva::erf(x); });
    }, "input"_a);
    m.def("erf", [](nb::ndarray<const std::complex<double>, nb::c_contig> input, double relerr) {
        return apply_func<std::complex<double>>(input, [&](std::complex<double> z) { return Faddeeva::erf(z, relerr); });
    }, "input"_a, "relerr"_a = 0);
    m.def("erf", [](double x) { return Faddeeva::erf(x); }, "x"_a);
    m.def("erf", [](std::complex<double> z, double relerr) { return Faddeeva::erf(z, relerr); }, "z"_a, "relerr"_a = 0);

    // ------------------------------------------------------------------------
    // erfc
    // ------------------------------------------------------------------------
    m.def("erfc", [](nb::ndarray<const double, nb::c_contig> input) {
        return apply_func<double>(input, [&](double x) { return Faddeeva::erfc(x); });
    }, "input"_a);
    m.def("erfc", [](nb::ndarray<const std::complex<double>, nb::c_contig> input, double relerr) {
        return apply_func<std::complex<double>>(input, [&](std::complex<double> z) { return Faddeeva::erfc(z, relerr); });
    }, "input"_a, "relerr"_a = 0);
    m.def("erfc", [](double x) { return Faddeeva::erfc(x); }, "x"_a);
    m.def("erfc", [](std::complex<double> z, double relerr) { return Faddeeva::erfc(z, relerr); }, "z"_a, "relerr"_a = 0);

    // ------------------------------------------------------------------------
    // erfi
    // ------------------------------------------------------------------------
    m.def("erfi", [](nb::ndarray<const double, nb::c_contig> input) {
        return apply_func<double>(input, [&](double x) { return Faddeeva::erfi(x); });
    }, "input"_a);
    m.def("erfi", [](nb::ndarray<const std::complex<double>, nb::c_contig> input, double relerr) {
        return apply_func<std::complex<double>>(input, [&](std::complex<double> z) { return Faddeeva::erfi(z, relerr); });
    }, "input"_a, "relerr"_a = 0);
    m.def("erfi", [](double x) { return Faddeeva::erfi(x); }, "x"_a);
    m.def("erfi", [](std::complex<double> z, double relerr) { return Faddeeva::erfi(z, relerr); }, "z"_a, "relerr"_a = 0);

    // ------------------------------------------------------------------------
    // erfcx
    // ------------------------------------------------------------------------
    m.def("erfcx", [](nb::ndarray<const double, nb::c_contig> input) {
        return apply_func<double>(input, [&](double x) { return Faddeeva::erfcx(x); });
    }, "input"_a);
    m.def("erfcx", [](nb::ndarray<const std::complex<double>, nb::c_contig> input, double relerr) {
        return apply_func<std::complex<double>>(input, [&](std::complex<double> z) { return Faddeeva::erfcx(z, relerr); });
    }, "input"_a, "relerr"_a = 0);
    m.def("erfcx", [](double x) { return Faddeeva::erfcx(x); }, "x"_a);
    m.def("erfcx", [](std::complex<double> z, double relerr) { return Faddeeva::erfcx(z, relerr); }, "z"_a, "relerr"_a = 0);

    // ------------------------------------------------------------------------
    // Dawson
    // ------------------------------------------------------------------------
    m.def("Dawson", [](nb::ndarray<const double, nb::c_contig> input) {
        return apply_func<double>(input, [&](double x) { return Faddeeva::Dawson(x); });
    }, "input"_a);
    m.def("Dawson", [](nb::ndarray<const std::complex<double>, nb::c_contig> input, double relerr) {
        return apply_func<std::complex<double>>(input, [&](std::complex<double> z) { return Faddeeva::Dawson(z, relerr); });
    }, "input"_a, "relerr"_a = 0);
    m.def("Dawson", [](double x) { return Faddeeva::Dawson(x); }, "x"_a);
    m.def("Dawson", [](std::complex<double> z, double relerr) { return Faddeeva::Dawson(z, relerr); }, "z"_a, "relerr"_a = 0);
}