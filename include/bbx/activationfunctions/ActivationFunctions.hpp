#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "AnyMovable.h"

namespace bbx {

template <class TBase>
class IAny : public TBase {
   public:
    virtual double evaluate(const double x) const = 0;

    virtual double derivative(const double x) const = 0;

    virtual Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const = 0;

    virtual Eigen::VectorXd derivative(const Eigen::VectorXd& x) const = 0;
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

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const override
    {
        return CBase::Object().evaluate(x);
    }

    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const override
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

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const;

    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const;
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

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const;

    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const;
};


// Implementation

inline Eigen::VectorXd Sigmoid::evaluate(const Eigen::VectorXd& x) const {
    Eigen::VectorXd z = x;

    for (double& i : z) {
        i = evaluate(i);
    }

    return z;
}

inline Eigen::VectorXd Sigmoid::derivative(const Eigen::VectorXd& x) const {
    Eigen::VectorXd z = x;

    for (double& i : z) {
        i = derivative(i);
    }

    return z;
}

inline Eigen::VectorXd Relu::evaluate(const Eigen::VectorXd& x) const {
    Eigen::VectorXd z = x;

    for (double& i : z) {
        i = evaluate(i);
    }

    return z;
}

inline Eigen::VectorXd Relu::derivative(const Eigen::VectorXd& x) const {
    Eigen::VectorXd z = x;

    for (double& i : z) {
        i = derivative(i);
    }

    return z;
}

}  // namespace bbx
