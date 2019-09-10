#ifndef _LAYER_SOFTMAX_HPP_
#define _LAYER_SOFTMAX_HPP_

#include "layer/Layer.hpp"
#include "math/Functions.hpp"

#include <iostream>

template <uint32_t in_act_num, uint32_t out_class_num>
class Softmax : public Layer<Vector<in_act_num>, Vector<out_class_num>> {
    typedef Vector<in_act_num> I;
    typedef Vector<out_class_num> O;
    typedef Vector<in_act_num + 1> XHat;
    typedef Eigen::Matrix<double, out_class_num, in_act_num + 1> Theta;

public:
    explicit Softmax(const Theta& theta = Theta::Random())
        : m_theta(theta)
    { }

    virtual ~Softmax() = default;

    virtual O infer(const I& input) const {
        XHat xHat;
        xHat << input, 1;
        O result = m_theta * xHat;
        Functions::softmax(result);
        return result;
    }

private:
    Theta m_theta;
};

#endif // !_LAYER_SOFTMAX_HPP_
