#pragma once

#include <bbx/types.hpp>
#include <cmath>
#include <random>

namespace bbx {

namespace detail {

inline std::mt19937& global_rng()
{
    static std::mt19937 gen{std::random_device{}()};
    return gen;
}

inline Matrix normal_random(Index rows, Index cols, double stddev)
{
    std::normal_distribution<double> dist(0.0, stddev);
    return Matrix::NullaryExpr(rows, cols, [&]() { return dist(global_rng()); });
}

}  // namespace detail

class UniformRandom {
   public:
    Matrix generate(Index rows, Index cols) const
    {
        return Matrix::Random(rows, cols);
    }
};

// Подходит для: Sigmoid, Tanh, Softplus, Softsign
class GlorotUniform {
   public:
    Matrix generate(Index rows, Index cols) const
    {
        double limit = std::sqrt(6.0 / (cols + rows));
        return Matrix::Random(rows, cols) * limit;
    }
};

// Подходит для: Sigmoid, Tanh
class GlorotNormal {
   public:
    Matrix generate(Index rows, Index cols) const
    {
        return detail::normal_random(rows, cols, std::sqrt(2.0 / (cols + rows)));
    }
};

// Подходит для: ReLU, LReLU, ELU, GELU, Swish
class HeNormal {
   public:
    Matrix generate(Index rows, Index cols) const
    {
        return detail::normal_random(rows, cols, std::sqrt(2.0 / cols));
    }
};

// Подходит для: ReLU, LReLU
class HeUniform {
   public:
    Matrix generate(Index rows, Index cols) const
    {
        double limit = std::sqrt(6.0 / cols);
        return Matrix::Random(rows, cols) * limit;
    }
};

// Обязательно для: SELU
// Подходит для: Sigmoid, Tanh
class LeCunNormal {
   public:
    Matrix generate(Index rows, Index cols) const
    {
        return detail::normal_random(rows, cols, std::sqrt(1.0 / cols));
    }
};

class Zeros {
   public:
    Matrix generate(Index rows, Index cols) const
    {
        return Matrix::Zero(rows, cols);
    }
};

}  // namespace bbx
