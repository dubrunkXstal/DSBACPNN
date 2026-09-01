#pragma once

#include <memory>
#include <bbx/types.hpp>

namespace bbx {

class AnyOptimizer {
private:
    struct Concept {
        virtual ~Concept() = default;

        virtual Matrix computeUpdate(const Matrix& gradient) = 0;

        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    template<class T>
    struct Model final : Concept {
        T object;

        explicit Model(T value) : object(std::move(value)) {}

        Matrix computeUpdate(const Matrix& gradient) override
        {
            return object.computeUpdate(gradient);
        }

        std::unique_ptr<Concept> clone() const override
        {
            return std::make_unique<Model<T> >(object);
        }
    };

public:
    AnyOptimizer() = default;

    template<class T>
    AnyOptimizer(T value) : object_(std::make_unique<Model<T> >(std::move(value))) {}

    AnyOptimizer(const AnyOptimizer& other)
        : object_(other.object_ ? other.object_->clone() : nullptr) {}

    AnyOptimizer& operator=(const AnyOptimizer& other)
    {
        if (this != &other) {
            object_ = other.object_ ? other.object_->clone() : nullptr;
        }

        return *this;
    }

    AnyOptimizer(AnyOptimizer&& other) noexcept = default;

    AnyOptimizer& operator=(AnyOptimizer&& other) noexcept = default;

    Matrix computeUpdate(const Matrix& gradient)
    {
        return object_->computeUpdate(gradient);
    }

private:
    std::unique_ptr<Concept> object_;
};

}  // namespace bbx
