#include <iostream>
#include <Eigen/Dense>

#include "types/Types.hpp"
#include "layer/Fc.hpp"
#include "layer/Softmax.hpp"

int main() {
    Eigen::Matrix<double, 4, 3 + 1> theta1;
    theta1 << .1, .2, .1, .2, .01, .02, .01, .02, .2, .1, .2, .1, .1, .05, .1, .05;
    Fc<3, 4> fc1{theta1};

    Eigen::Matrix<double, 3, 4 + 1> theta2;
    theta2 << .1, .2, .1, .2, .1, .05, .02, .05, .02, .05, .1, .02, .1, .02, .1;
    Softmax<4, 3> sm{theta2};

    Vector<3> i;
    i << 1, 2, 3;

    Vector<4> h1 = fc1.infer(i);
    std::cout << "h1:\n" << h1 << '\n';
    Vector<3> y  = sm.infer(h1);
    std::cout << "y:\n" << y << '\n';
    return 0;
}
