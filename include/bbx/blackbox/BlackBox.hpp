#pragma once

#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <bbx/activationfunctions/ActivationFunctions.hpp>
#include <bbx/types.hpp>

namespace bbx {

inline constexpr double gradient_step = 0.01;

class LossFunction {
public:
    double distance(const Vector& z, const Vector& y) const
    {
        return pow((z - y).norm(), 2);
    }

    RowVector gradient(const Vector& z, const Vector& y) const;
};

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

class BlackBox {
public:
    BlackBox(std::ifstream& settings);

    Vector evaluate(const Vector& x) const;

    void tuning(const Vector& x, const Vector& y);

private:
    size_t blocks_cnt;
    std::vector<std::unique_ptr<Block> > blocks;
    LossFunction loss;
};


// Implementation

inline RowVector LossFunction::gradient(const Vector& z, const Vector& y) const {
    RowVector result(z.rows());

    for (int i = 0; i < z.rows(); ++i) {
        result[i] = 2 * (z[i] - y[i]);
    }

    return result;
}

inline BlackBox::BlackBox(std::ifstream& settings) : blocks(std::vector<std::unique_ptr<Block> >()) {
    Index in_dim;
    Index out_dim;
    std::string activaton;
    std::string line;

    std::getline(settings, line);
    std::stringstream ss(line);
    ss >> blocks_cnt;

    for (int i = 0; i < blocks_cnt; ++i) {
        getline(settings, line);
        ss = std::stringstream(line);
        ss >> in_dim >> out_dim >> activaton;

        if (activaton == "sigmoid") {
            blocks.emplace_back(std::make_unique<Block>(in_dim, out_dim, Sigmoid()));
        } else if (activaton == "relu") {
            blocks.emplace_back(std::make_unique<Block>(in_dim, out_dim, Relu()));
        } else {
            throw std::runtime_error("Didn't found activaton function for the block.");
        }
    }
}

inline Vector BlackBox::evaluate(const Vector& x) const {
    Vector result = x;

    for (int i = 0; i < blocks_cnt; ++i) {
        result = blocks[i]->evaluate(result);
    }

    return result;
}

inline void BlackBox::tuning(const Vector& x, const Vector& y) {
    std::vector<std::unique_ptr<Vector> > remember_output;
    remember_output.emplace_back(std::make_unique<Vector>(blocks[0]->evaluate(x)));

    for (int i = 1; i < blocks_cnt; ++i) {
        remember_output.emplace_back(
            std::make_unique<Vector>(blocks[i]->evaluate(*remember_output[i - 1])));
    }

    RowVector u = loss.gradient(*(remember_output[blocks_cnt - 1]), y);
    RowVector u_next;

    for (int i = blocks_cnt - 1; i > 0; --i) {
        u_next = blocks[i]->propogateBack(*(remember_output[i - 1]), u);
        blocks[i]->gradientDescent(*(remember_output[i - 1]), u);
        u = u_next;
    }

    blocks[0]->gradientDescent(x, u);
}

}  // namespace bbx
