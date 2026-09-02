#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <bbx/types.hpp>

namespace bbx {

class AnyLoss {
   private:
    struct Concept {
        virtual ~Concept() = default;

        virtual double distance(const Vector& z, const Vector& y) const = 0;

        virtual RowVector gradient(const Vector& z, const Vector& y) const = 0;

        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    template <class T>
    struct Model final : Concept {
        T object;

        explicit Model(T value) : object(std::move(value)) {}

        double distance(const Vector& z, const Vector& y) const override
        {
            return object.distance(z, y);
        }

        RowVector gradient(const Vector& z, const Vector& y) const override
        {
            return object.gradient(z, y);
        }

        std::unique_ptr<Concept> clone() const override
        {
            return std::make_unique<Model<T> >(object);
        }
    };

   public:
    AnyLoss() = default;

    template <class T>
    AnyLoss(T value) : object_(std::make_unique<Model<T> >(std::move(value)))
    {}

    AnyLoss(const AnyLoss& other) : object_(other.object_ ? other.object_->clone() : nullptr) {}

    AnyLoss& operator=(const AnyLoss& other)
    {
        if (this != &other) {
            object_ = other.object_ ? other.object_->clone() : nullptr;
        }

        return *this;
    }

    AnyLoss(AnyLoss&& other) noexcept = default;

    AnyLoss& operator=(AnyLoss&& other) noexcept = default;

    double distance(const Vector& z, const Vector& y) const
    {
        return object_->distance(z, y);
    }

    RowVector gradient(const Vector& z, const Vector& y) const
    {
        return object_->gradient(z, y);
    }

   private:
    std::unique_ptr<Concept> object_;
};

}  // namespace bbx
