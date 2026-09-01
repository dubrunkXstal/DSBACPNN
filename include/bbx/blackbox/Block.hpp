#pragma once

#include <bbx/activationfunctions/AnyActivationFunction.hpp>
#include <bbx/types.hpp>

namespace bbx {

struct BlockConfig {
    Index input_dimension;
    Index output_dimension;
    AnyActivation activation_function;
};

inline constexpr double gradient_step = 0.01;

class Block {
public:
    Block(Index in_dim, Index out_dim, const AnyActivation& sigma)
      : in_dim(in_dim),
        out_dim(out_dim),
        A(Matrix::Random(out_dim, in_dim)),
        b(Vector::Random(out_dim)),
        sigma(sigma) {}

    Vector evaluate(const Vector& x) const
    {
        return sigma.evaluate(A * x + b);
    }

    Matrix evaluate(const Matrix& x_batch) const
    {
        return ((A * x_batch).colwise() + b).unaryExpr([this](double v) {
            return sigma.evaluate(v);
        });
    }

    Matrix grad_A(const Vector& x, const RowVector& u) const
    {
        Vector delta = sigma.derivative(A * x + b).array() * u.transpose().array();

        return delta * x.transpose();
    }

    Matrix grad_A(const Matrix& x_batch, const Matrix& u_batch) const
    {
        Matrix Z = (A * x_batch).colwise() + b;

        Matrix Delta = Z.unaryExpr([this](double v) {
            return sigma.derivative(v);
        });

        Delta.array() *= u_batch.transpose().array();

        return Delta * x_batch.transpose();
    }

    Vector grad_b(const Vector& x, const RowVector& u) const
    {
        return sigma.derivative(A * x + b).asDiagonal() * u.transpose();
    }

    Vector grad_b(const Matrix& x_batch, const Matrix& u_batch) const
    {
        Matrix Z = (A * x_batch).colwise() + b;

        Matrix Delta = Z.unaryExpr([this](double v) {
            return sigma.derivative(v);
        });

        Delta.array() *= u_batch.transpose().array();

        return Delta.rowwise().sum();
    }

    void gradientDescent(const Vector& x, const RowVector& u)
    {
        A -= grad_A(x, u) * gradient_step;
        b -= grad_b(x, u) * gradient_step;
    }

    void gradientDescent(const Matrix& x_batch, const Matrix& u_batch)
    {
        A -= grad_A(x_batch, u_batch) * gradient_step;
        b -= grad_b(x_batch, u_batch) * gradient_step;
    }

    RowVector propogateBack(const Vector& x, const RowVector& u) const
    {
        return u * sigma.derivative(A * x + b).asDiagonal() * A;
    }

    Matrix propogateBack(const Matrix& x_batch, const Matrix& u_batch) const
    {
        Matrix Z = (A * x_batch).colwise() + b;

        Matrix Delta = Z.unaryExpr([this](double v) {
            return sigma.derivative(v);
        });

        Delta.array() *= u_batch.transpose().array();

        return Delta.transpose() * A;
    }

private:
    Index in_dim;
    Index out_dim;
    Matrix A;
    Vector b;
    AnyActivation sigma;
};

}  // namespace bbx
