#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <bbx/types.hpp>

namespace bbx {

struct Relu {
    double evaluate(const double x) const { return x < 0 ? 0 : x; }

    double derivative(const double x) const { return x < 0 ? 0 : 1; }

    Vector evaluate(const Vector& x) const;

    Vector derivative(const Vector& x) const;
};

struct Sigmoid {
    double evaluate(const double x) const { return 1 / (1 + exp(-x)); }

    double derivative(const double x) const {
        double s = evaluate(x);
        return s * (1.0 - s);
    }

    Vector evaluate(const Vector& x) const;

    Vector derivative(const Vector& x) const;
};

// Implementation

inline Vector Sigmoid::evaluate(const Vector& x) const {
    Vector z = x;

    for (double& i : z) {
        i = evaluate(i);
    }

    return z;
}

inline Vector Sigmoid::derivative(const Vector& x) const {
    Vector z = x;

    for (double& i : z) {
        i = derivative(i);
    }

    return z;
}

inline Vector Relu::evaluate(const Vector& x) const {
    Vector z = x;

    for (double& i : z) {
        i = evaluate(i);
    }

    return z;
}

inline Vector Relu::derivative(const Vector& x) const {
    Vector z = x;

    for (double& i : z) {
        i = derivative(i);
    }

    return z;
}

}  // namespace bbx
