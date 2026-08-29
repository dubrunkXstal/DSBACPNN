#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <bbx/types.hpp>
#include <bbx/activationfunctions/AnyActivationFunction.hpp>

namespace bbx {

struct Relu {
    Relu() = default;

    Relu(const Relu& other) = delete;

    Relu(Relu&& other) noexcept = default;

    Relu& operator=(const Relu& other) = delete;

    Relu& operator=(Relu&& other) noexcept = default;

    ~Relu() = default;

    double evaluate(const double x) const
    {
        if (x < 0) {
            return 0;
        }

        return x;
    }

    double derivative(const double x) const
    {
        if (x < 0) {
            return 0;
        }

        return 1;
    }

    Vector evaluate(const Vector& x) const;

    Vector derivative(const Vector& x) const;
};

struct Sigmoid {
    Sigmoid() = default;

    Sigmoid(const Sigmoid& other) = delete;

    Sigmoid(Sigmoid&& other) noexcept = default;

    Sigmoid& operator=(const Sigmoid& other) = delete;

    Sigmoid& operator=(Sigmoid&& other) noexcept = default;

    ~Sigmoid() = default;

    double evaluate(const double x) const
    {
        return 1 / (1 + exp(-x));
    }

    double derivative(const double x) const
    {
        return exp(-x) / pow(1 + exp(-x), 2);
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
