#include <vector>
#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Efficient Attention Implementation
class EfficientAttention {
public:
    EfficientAttention(int dim_head) : dim_head(dim_head) {}

    // Forward pass: Efficient Attention
    inline std::vector<std::vector<double>> forward(
        const std::vector<std::vector<double>>& queries,
        const std::vector<std::vector<double>>& keys,
        const std::vector<std::vector<double>>& values) {

        int seq_len = queries.size();
        int head_dim = queries[0].size();

        // Compute the normalization factor
        std::vector<double> norm_factors(seq_len, 0.0);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                norm_factors[i] += exp(dot_product(queries[i], keys[j]));
            }
        }

        // Compute attention output
        std::vector<std::vector<double>> output(seq_len, std::vector<double>(head_dim, 0.0));
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                double weight = exp(dot_product(queries[i], keys[j])) / norm_factors[i];
                for (int d = 0; d < head_dim; ++d) {
                    output[i][d] += weight * values[j][d];
                }
            }
        }

        return output;
    }

private:
    int dim_head;

    // Inline function for dot product
    inline double dot_product(const std::vector<double>& a, const std::vector<double>& b) const {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
};

PYBIND11_MODULE(efficient_attention, m) {
    py::class_<EfficientAttention>(m, "EfficientAttention")
        .def(py::init<int>())
        .def("forward", &EfficientAttention::forward);
}
