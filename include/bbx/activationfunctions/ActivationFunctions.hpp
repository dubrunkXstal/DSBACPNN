#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <bbx/types.hpp>
#include "AnyMovable.h"

namespace bbx {

template <class TBase>
class IAny : public TBase {
   public:
    virtual double evaluate(const double x) const = 0;

    virtual double derivative(const double x) const = 0;

    virtual Vector evaluate(const Vector& x) const = 0;

    virtual Vector derivative(const Vector& x) const = 0;
};

template <class TBase, class TObject>
class CAnyImpl : public TBase {
    using CBase = TBase;

   public:
    using CBase::CBase;

    double evaluate(const double x) const override
    {
        return CBase::Object().evaluate(x);
    }

    double derivative(const double x) const override
    {
        return CBase::Object().derivative(x);
    }

    Vector evaluate(const Vector& x) const override
    {
        return CBase::Object().evaluate(x);
    }

    Vector derivative(const Vector& x) const override
    {
        return CBase::Object().derivative(x);
    }
};

class CAny : public NSLibrary::CAnyMovable<IAny, CAnyImpl> {
    using CBase = CAnyMovable<IAny, CAnyImpl>;

   public:
    using CBase::CBase;
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
