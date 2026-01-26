#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include "AnyMovable.h"


template<class TBase>
class IAny : public TBase {
public:
    virtual double evaluate(const double x) const = 0;

    virtual double derivative(const double x) const = 0;

    virtual Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const = 0;

    virtual Eigen::VectorXd derivative(const Eigen::VectorXd& x) const = 0;
};


template<class TBase, class TObject>
class CAnyImpl : public TBase {
    using CBase = TBase;
public:
    using CBase::CBase;

    double evaluate(const double x) const override {
        return CBase::Object().evaluate(x);
    }

    double derivative(const double x) const override {
        return CBase::Object().derivative(x);
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const override {
        return CBase::Object().evaluate(x);
    }

    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const override {
        return CBase::Object().derivative(x);
    }
};


class CAny : public NSLibrary::CAnyMovable<IAny, CAnyImpl> {
    using CBase = CAnyMovable<IAny, CAnyImpl>;
public:
    using CBase::CBase;
};


struct Sigmoid {
    Sigmoid();

    Sigmoid(const Sigmoid& other) = delete;

    Sigmoid(Sigmoid&& other) noexcept;

    Sigmoid& operator=(const Sigmoid& other) = delete;

    Sigmoid& operator=(Sigmoid&& other) noexcept;

    ~Sigmoid();


    double evaluate(const double x) const;

    double derivative(const double x) const;

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const;

    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const;
};


struct Relu {
    Relu();

    Relu(const Relu& other) = delete;

    Relu(Relu&& other) noexcept;

    Relu& operator=(const Relu& other) = delete;

    Relu& operator=(Relu&& other) noexcept;

    ~Relu();


    double evaluate(const double x) const;

    double derivative(const double x) const;

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const;

    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const;
};

#include "ActivationFunctions_impl.cpp"

#endif  // ACTIVATIONFUNCTIONS_H
