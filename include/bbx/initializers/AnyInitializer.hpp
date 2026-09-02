#pragma once

#include <bbx/core/types.hpp>
#include <memory>

namespace bbx {

class AnyInitializer {
   private:
    struct Concept {
        virtual ~Concept() = default;

        virtual Matrix generate(Index rows, Index cols) const = 0;

        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    template <class T>
    struct Model final : Concept {
        T object;

        explicit Model(T value) : object(std::move(value)) {}

        Matrix generate(Index rows, Index cols) const override
        {
            return object.generate(rows, cols);
        }

        std::unique_ptr<Concept> clone() const override
        {
            return std::make_unique<Model<T>>(object);
        }
    };

   public:
    AnyInitializer() = default;

    template <class T>
    AnyInitializer(T value) : object_(std::make_unique<Model<T>>(std::move(value)))
    {}

    AnyInitializer(const AnyInitializer& other) : object_(other.object_ ? other.object_->clone() : nullptr) {}

    AnyInitializer& operator=(const AnyInitializer& other)
    {
        if (this != &other) {
            object_ = other.object_ ? other.object_->clone() : nullptr;
        }
        return *this;
    }

    AnyInitializer(AnyInitializer&&) noexcept = default;

    AnyInitializer& operator=(AnyInitializer&&) noexcept = default;

    Matrix generate(Index rows, Index cols) const
    {
        return object_->generate(rows, cols);
    }

   private:
    std::unique_ptr<Concept> object_;
};

}  // namespace bbx
