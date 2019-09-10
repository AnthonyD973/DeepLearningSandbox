#include <iostream>
#include <Eigen/Dense>

#include "types/Types.hpp"
#include "layer/Fc.hpp"

int main() {
    Eigen::Matrix<double, 4, 3 + 1> theta;
    theta << .1, .2, .1, .2, .01, .02, .01, .02, .2, .1, .2, .1, .1, .05, .1, .05;
    Fc<3, 4> fc1{theta};
    Vector<3> i;
    i << 1, 2, 3;
    
    Vector<4> o = fc1.infer(i);
    std::cout << o << '\n';
    return 0;
}
