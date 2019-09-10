#include <math.h>

namespace Functions {
    inline double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }
}
