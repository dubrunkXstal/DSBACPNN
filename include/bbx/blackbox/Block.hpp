#include <bbx/activationfunctions/AnyActivationFunction.hpp>
#include <bbx/types.hpp>

namespace bbx {

struct BlockConfig {
    Index input_dimension;
    Index output_dimension;
    CAny activation_function;
};

inline constexpr double gradient_step = 0.01;

class Block {
public:
    Block(Index in_dim, Index out_dim, CAny&& sigma)
      : in_dim(in_dim),
        out_dim(out_dim),
        A(Matrix::Random(out_dim, in_dim)),
        b(Matrix::Random(out_dim, 1)),
        sigma(std::move(sigma)) {}

    Vector evaluate(const Vector& x) const
    {
        return sigma->evaluate(A * x + b);
    }

    Matrix grad_A(const Vector& x, const RowVector& u) const
    {
        return sigma->derivative(A * x + b).asDiagonal() * u.transpose() * x.transpose();
    }

    Vector grad_b(const Vector& x, const RowVector& u) const
    {
        return sigma->derivative(A * x + b).asDiagonal() * u.transpose();
    }

    void gradientDescent(const Vector& x, const RowVector& u)
    {
        A -= grad_A(x, u) * gradient_step;
        b -= grad_b(x, u) * gradient_step;
    }

    RowVector propogateBack(const Vector& x, const RowVector& u) const
    {
        return u * sigma->derivative(A * x + b).asDiagonal() * A;
    }

private:
    Index in_dim;
    Index out_dim;
    Matrix A;
    Vector b;
    CAny sigma;
};

}  // namespace bbx
