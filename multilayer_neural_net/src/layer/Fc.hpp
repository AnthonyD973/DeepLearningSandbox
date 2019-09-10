#ifndef _LAYER_FC_HPP_
#define _LAYER_FC_HPP_

#include <inttypes.h>

#include "layer/Layer.hpp"
#include "math/Functions.hpp"

template <uint32_t in_act_num, uint32_t out_act_num>
class Fc : public Layer<Vector<in_act_num>, Vector<out_act_num>> {
    typedef Vector<in_act_num> X;
    typedef Vector<in_act_num + 1> XHat;
    typedef Vector<out_act_num> O;
    typedef Eigen::Matrix<double, out_act_num, in_act_num + 1> Theta;

private:
    static XHat extendInput(const X& x) {
        XHat xHat;
        xHat << x, 1;
        return xHat;
    }

public:
    explicit Fc(const Theta& theta = Theta::Random())
        : m_theta(theta)
    { }

    virtual ~Fc() = default;

    virtual O infer(const X& input) const {
        O o = preactivate(extendInput(input));
        activate(o);
        return o;
    }

protected:
    inline O preactivate(const XHat& xHat) const {
        return m_theta * xHat;
    }

    inline void activate(O& o) const {
        // Not using matrix multiplication ; that would take way
        // longer on a CPU.
        for (size_t i = 0; i < o.rows(); ++i) {
            auto& oi = o(i);
            oi = Functions::sigmoid(oi);
        }
    }

private:
    Theta m_theta;
};

#endif // !_LAYER_FC_HPP_
