#ifndef _MATH_FUNCTIONS_HPP_
#define _MATH_FUNCTIONS_HPP_

#include <math.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <type_traits>

#include "types/Types.hpp"

namespace Functions {
    template <class T>
    inline T sigmoid(T x) {
        return 1 / (1 + exp(-x));
    }

    template <class T>
    inline T softmax(T& input) {
        typedef typename std::remove_reference<decltype(input(0))>::type elem_t;

        // Compute exponential of inputs.
        std::vector<elem_t> exponentiated;
        for (size_t i = 0; i < input.rows(); ++i) {
            exponentiated.push_back(exp(input(i)));
        }

        // Compute normalization factor.
        elem_t normalFactor =
            std::accumulate(exponentiated.begin(), exponentiated.end(), static_cast<elem_t>(0));

        std::cout << normalFactor << std::endl;

        // Apply softmax.
        size_t i = 0;
        std::for_each(
            exponentiated.begin(),
            exponentiated.end(),
            [&input, &i, normalFactor](elem_t expElem) {
                input(i) = expElem / normalFactor;
                ++i;
            }
        );
    }
}

#endif // !_MATH_FUNCTIONS_HPP_
