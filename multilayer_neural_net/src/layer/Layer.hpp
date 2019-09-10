#ifndef _LAYER_LAYER_HPP_
#define _LAYER_LAYER_HPP_

#include "types/Types.hpp"

template <class Input, class Output>
class Layer {
public:
    explicit Layer() = default;
    virtual ~Layer() = default;

    virtual Output infer(const Input& input) const = 0;
};

#endif // !_LAYER_LAYER_HPP_
