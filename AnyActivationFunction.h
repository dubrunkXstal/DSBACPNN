// #include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "AnyMovable.h"



template<class TBase>
class IAny : public TBase {
public:
    // virtual void print() const = 0;

    virtual double evaluate(const double x) const = 0;

    virtual double derivative(const double x) const = 0;

    virtual Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const = 0;

    virtual Eigen::MatrixXd derivative(const Eigen::VectorXd& x) const = 0;
};


template<class TBase, class TObject>
class CAnyImpl : public TBase {
    using CBase = TBase;
public:
    using CBase::CBase;
    // void print() const override {
    //     std::cout << "data = " << CBase::Object() << std::endl;
    // }

    double evaluate(const double x) const override {
        return CBase::Object().evaluate(x);
    }

    double derivative(const double x) const override {
        return CBase::Object().derivative(x);
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const override {
        return CBase::Object().evaluate(x);
    }

    Eigen::MatrixXd derivative(const Eigen::VectorXd& x) const override {
        return CBase::Object().derivative(x);
    }
};


class CAny : public NSLibrary::CAnyMovable<IAny, CAnyImpl> {
    using CBase = CAnyMovable<IAny, CAnyImpl>;
public:
    using CBase::CBase;
    // friend bool operator==(const CAny&, const CAny&) {
    // ...
    // }
};
