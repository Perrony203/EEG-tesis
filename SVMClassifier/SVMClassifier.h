#pragma once
#include <cstdarg>
#include <cstdint>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class SVM {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        float kernels[4] = { 0 };
                        float decisions[1] = { 0 };
                        int votes[2] = { 0 };
                        kernels[0] = compute_kernel(x,   3.4  , 1.9  , 0.2 );
                        kernels[1] = compute_kernel(x,   2.3  , 1.3  , 0.3 );
                        kernels[2] = compute_kernel(x,   3.3  , 1.7  , 0.5 );
                        kernels[3] = compute_kernel(x,   2.5  , 3.0  , 1.1 );
                        float decision = -1.299839158621;
                        decision = decision - ( + kernels[0] * -0.095953663262  + kernels[1] * -0.109615636154  + kernels[2] * -0.544583702947 );
                        decision = decision - ( + kernels[3] * 0.750153002363 );

                        return decision > 0 ? 0 : 1;
                    }

                protected:
                    /**
                    * Compute kernel between feature vector and support vector.
                    * Kernel type: linear
                    */
                    float compute_kernel(float *x, ...) {
                        va_list w;
                        va_start(w, 3);
                        float kernel = 0.0;

                        for (uint16_t i = 0; i < 3; i++) {
                            kernel += x[i] * va_arg(w, double);
                        }

                        return kernel;
                    }
                };
            }
        }
    }