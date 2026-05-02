#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// =============================================================================
// ALGORITHM SECTION
// =============================================================================

float my_logf(float a) {
    float i, m, r, s, t;
    int e;
    m = frexpf(a, &e);
    if (m < 0.666666667f) {
        m = m + m;
        e = e - 1;
    }
    i = (float)e;
    m = m - 1.0f;
    s = m * m;
    r =             -0.130310059f;
    t =              0.140869141f;
    r = fmaf(r, s, -0.121484190f);
    t = fmaf(t, s,  0.139814854f);
    r = fmaf(r, s, -0.166846052f);
    t = fmaf(t, s,  0.200120345f);
    r = fmaf(r, s, -0.249996200f);
    r = fmaf(t, m, r);
    r = fmaf(r, m,  0.333331972f);
    r = fmaf(r, m, -0.500000000f);
    r = fmaf(r, s, m);
    r = fmaf(i,  0.693147182f, r);
    if (!((a > 0.0f) && (a <= 3.40282346e+38f))) {
        r = a + a;
        if (a  < 0.0f) {
            float zero = 0.0f;
            r = zero / zero;
        }
        if (a == 0.0f) {
            float one = 1.0f;
            float zero = 0.0f;
            r = -one / zero;
        }
    }
    return r;
}

float my_erfinvf(float a) {
    float p, t;
    t = fmaf(a, 0.0f - a, 1.0f);
    t = my_logf(t);
    if (std::abs(t) > 6.125f) {
        p =              3.03697567e-10f;
        p = fmaf(p, t,   2.93243101e-8f);
        p = fmaf(p, t,   1.22150334e-6f);
        p = fmaf(p, t,   2.84108955e-5f);
        p = fmaf(p, t,   3.93552968e-4f);
        p = fmaf(p, t,   3.02698812e-3f);
        p = fmaf(p, t,   4.83185798e-3f);
        p = fmaf(p, t, -2.64646143e-1f);
        p = fmaf(p, t,   8.40016484e-1f);
    } else {
        p =              5.43877832e-9f;
        p = fmaf(p, t,   1.43285448e-7f);
        p = fmaf(p, t,   1.22774793e-6f);
        p = fmaf(p, t,   1.12963626e-7f);
        p = fmaf(p, t, -5.61530760e-5f);
        p = fmaf(p, t, -1.47697632e-4f);
        p = fmaf(p, t,   2.31468678e-3f);
        p = fmaf(p, t,   1.15392581e-2f);
        p = fmaf(p, t, -2.32015476e-1f);
        p = fmaf(p, t,   8.86226892e-1f);
    }
    return a * p;
}

// =============================================================================
// NANOBIND WRAPPER
// =============================================================================

NB_MODULE(_erfinv, m) {
    m.def("erfinv", [](nb::ndarray<double, nb::c_contig> input) {
        std::vector<size_t> shape(input.ndim());
        for (size_t i = 0; i < input.ndim(); ++i) shape[i] = input.shape(i);
        
        double* data = new double[input.size()];
        nb::capsule owner(data, [](void *p) noexcept { delete[] (double *) p; });
        nb::ndarray<nb::numpy, double, nb::c_contig> out_arr(data, input.ndim(), shape.data(), owner);
        
        auto in_ptr = input.data();
        auto out_ptr = out_arr.data();
        
        for (size_t i = 0; i < input.size(); ++i) {
            out_ptr[i] = (double)my_erfinvf((float)in_ptr[i]);
        }
        
        return out_arr;
    }, "input"_a, "Calculate the inverse error function, erfinv(x).");

    m.def("erfinv", [](double x) {
        return (double)my_erfinvf((float)x);
    }, "x"_a, "Calculate the inverse error function, erfinv(x).");
}
