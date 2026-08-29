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

}  // namespace bbx
