#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <bbx/types.hpp>

namespace bbx {

class AnyActivation {
   private:
    struct Concept {
        virtual ~Concept() = default;

        virtual double evaluate(const double x) const = 0;
        virtual double derivative(const double x) const = 0;

        virtual Vector evaluate(const Vector& x) const = 0;
        virtual Vector derivative(const Vector& x) const = 0;

        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    template <class T>
    struct Model final : Concept {
        T object;

        explicit Model(T value) : object(std::move(value)) {}

        double evaluate(const double x) const override
        {
            return object.evaluate(x);
        }

        double derivative(const double x) const override
        {
            return object.derivative(x);
        }

        Vector evaluate(const Vector& x) const override
        {
            return object.evaluate(x);
        }

        Vector derivative(const Vector& x) const override
        {
            return object.derivative(x);
        }

        std::unique_ptr<Concept> clone() const override
        {
            return std::make_unique<Model<T> >(object);
        }
    };

   public:
    AnyActivation() = default;

    template <class T>
    AnyActivation(T value) : object_(std::make_unique<Model<T> >(std::move(value)))
    {}

    AnyActivation(const AnyActivation& other) : object_(other.object_ ? other.object_->clone() : nullptr) {}

    AnyActivation& operator=(const AnyActivation& other)
    {
        if (this != &other) {
            object_ = other.object_ ? other.object_->clone() : nullptr;
        }

        return *this;
    }

    AnyActivation(AnyActivation&& other) noexcept = default;

    AnyActivation& operator=(AnyActivation&& other) noexcept = default;

    double evaluate(const double x) const
    {
        return object_->evaluate(x);
    }

    double derivative(const double x) const
    {
        return object_->derivative(x);
    }

    Vector evaluate(const Vector& x) const
    {
        return object_->evaluate(x);
    }

    Vector derivative(const Vector& x) const
    {
        return object_->derivative(x);
    }

   private:
    std::unique_ptr<Concept> object_;
};

}  // namespace bbx
