#include <math.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "types/Types.hpp"

namespace Functions {
    template <class T>
    inline T sigmoid(T x) {
        return 1 / (1 + exp(-x));
    }

    template <uint32_t classNum>
    inline Vector<classNum> softmax(const Vector<classNum>& input) {
        typedef decltype(input(0)) elem_t;

        // Compute exponential of inputs.
        std::vector<elem_t> exponentiated;
        for (size_t i = 0; i < input.rows(); ++i) {
            exponentiated.push_back(exp(input(i)));
        }

        // Compute normalization factor.
        elem_t normalFactor =
            std::accumulate(exponentiated.begin(), exponentiated.end(), 0);

        // Apply softmax.
        Vector<classNum> softmaxResult;
        size_t i = 0;
        std::for_each(
            exponentiated.begin(),
            exponentiated.end(),
            [&softmaxResult, &i, normalFactor](auto expElem) {
                softmaxResult(i) = expElem / normalFactor;
                ++i;
            }
        );

        return softmaxResult;
    }
}
