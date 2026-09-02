#pragma once

#include <bbx/activationfunctions/AnyActivationFunction.hpp>
#include <bbx/optimizers/AnyOptimizer.hpp>
#include <bbx/optimizers/Optimizers.hpp>
#include <bbx/types.hpp>

namespace bbx {

struct BlockConfig {
    Index input_dimension;
    Index output_dimension;
    AnyActivation activation_function;
    AnyOptimizer optimizer;

    BlockConfig(Index input_dimension, Index output_dimension, const AnyActivation& activation_function,
                const AnyOptimizer& optimizer = VanillaDescent{})
        : input_dimension(input_dimension),
          output_dimension(output_dimension),
          activation_function(activation_function),
          optimizer(optimizer)
    {}
};

class Block {
   public:
    Block(Index input_dimension, Index output_dimension, const AnyActivation& sigma,
          const AnyOptimizer& optimizer = VanillaDescent{})
        : input_dimension_(input_dimension),
          output_dimension_(output_dimension),
          A_(Matrix::Random(output_dimension, input_dimension)),
          b_(Vector::Random(output_dimension)),
          sigma_(sigma),
          A_optimizer_(optimizer),
          b_optimizer_(optimizer)
    {}

    Vector evaluate(const Vector& x) const
    {
        return sigma_.evaluate(A_ * x + b_);
    }

    Matrix evaluate(const Matrix& x_batch) const
    {
        return ((A_ * x_batch).colwise() + b_).unaryExpr([this](double v) { return sigma_.evaluate(v); });
    }

    Matrix grad_A(const Vector& x, const RowVector& u) const
    {
        Vector delta = sigma_.derivative(A_ * x + b_).array() * u.transpose().array();
        return delta * x.transpose();
    }

    Matrix grad_A(const Matrix& x_batch, const Matrix& u_batch) const;

    Vector grad_b(const Vector& x, const RowVector& u) const
    {
        return sigma_.derivative(A_ * x + b_).asDiagonal() * u.transpose();
    }

    Vector grad_b(const Matrix& x_batch, const Matrix& u_batch) const;

    void gradientDescent(const Vector& x, const RowVector& u)
    {
        A_ += A_optimizer_.computeUpdate(grad_A(x, u));
        b_ += b_optimizer_.computeUpdate(grad_b(x, u));
    }

    void gradientDescent(const Matrix& x_batch, const Matrix& u_batch)
    {
        A_ += A_optimizer_.computeUpdate(grad_A(x_batch, u_batch));
        b_ += b_optimizer_.computeUpdate(grad_b(x_batch, u_batch));
    }

    RowVector propogateBack(const Vector& x, const RowVector& u) const
    {
        return u * sigma_.derivative(A_ * x + b_).asDiagonal() * A_;
    }

    Matrix propogateBack(const Matrix& x_batch, const Matrix& u_batch) const;

    void setActivation(const AnyActivation sigma)
    {
        sigma_ = sigma;
    }

    void setOptimizer(const AnyOptimizer& optimizer)
    {
        A_optimizer_ = optimizer;
        b_optimizer_ = optimizer;
    }

   private:
    Index input_dimension_;
    Index output_dimension_;

    Matrix A_;
    Vector b_;
    AnyActivation sigma_;

    AnyOptimizer A_optimizer_;
    AnyOptimizer b_optimizer_;
};

// Implementation

inline Matrix Block::grad_A(const Matrix& x_batch, const Matrix& u_batch) const
{
    Matrix Z = (A_ * x_batch).colwise() + b_;

    Matrix Delta = Z.unaryExpr([this](double v) { return sigma_.derivative(v); });

    Delta.array() *= u_batch.transpose().array();

    return Delta * x_batch.transpose();
}

inline Vector Block::grad_b(const Matrix& x_batch, const Matrix& u_batch) const
{
    Matrix Z = (A_ * x_batch).colwise() + b_;

    Matrix Delta = Z.unaryExpr([this](double v) { return sigma_.derivative(v); });

    Delta.array() *= u_batch.transpose().array();

    return Delta.rowwise().sum();
}

inline Matrix Block::propogateBack(const Matrix& x_batch, const Matrix& u_batch) const
{
    Matrix Z = (A_ * x_batch).colwise() + b_;

    Matrix Delta = Z.unaryExpr([this](double v) { return sigma_.derivative(v); });

    Delta.array() *= u_batch.transpose().array();

    return Delta.transpose() * A_;
}

}  // namespace bbx
