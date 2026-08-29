#pragma once

#include <cstdio>
#include <fstream>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <bbx/activationfunctions/ActivationFunctions.hpp>

namespace bbx {

inline constexpr double gradient_step = 0.01;

class LossFunction {
public:
    double distance(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const
    {
        return pow((z - y).norm(), 2);
    }

    Eigen::RowVectorXd gradient(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const;
};

class Block {
public:
    Block(size_t in_dim, size_t out_dim, CAny&& sigma)
      : in_dim(in_dim),
        out_dim(out_dim),
        A(Eigen::MatrixXd::Random(out_dim, in_dim)),
        b(Eigen::MatrixXd::Random(out_dim, 1)),
        sigma(std::move(sigma)) {}

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const
    {
        return sigma->evaluate(A * x + b);
    }

    Eigen::MatrixXd grad_A(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
    {
        return sigma->derivative(A * x + b).asDiagonal() * u.transpose() * x.transpose();
    }

    Eigen::VectorXd grad_b(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
    {
        return sigma->derivative(A * x + b).asDiagonal() * u.transpose();
    }

    void gradientDescent(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u)
    {
        A -= grad_A(x, u) * gradient_step;
        b -= grad_b(x, u) * gradient_step;
    }

    Eigen::RowVectorXd propogateBack(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
    {
        return u * sigma->derivative(A * x + b).asDiagonal() * A;
    }

private:
    size_t in_dim;
    size_t out_dim;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    CAny sigma;
};

class BlackBox {
public:
    BlackBox(std::ifstream& settings);

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const;

    void tuning(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

private:
    size_t blocks_cnt;
    std::vector<std::unique_ptr<Block> > blocks;
    LossFunction loss;
};


// Implementation

inline Eigen::RowVectorXd LossFunction::gradient(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const {
    Eigen::RowVectorXd result(z.rows());

    for (int i = 0; i < z.rows(); ++i) {
        result[i] = 2 * (z[i] - y[i]);
    }

    return result;
}

inline BlackBox::BlackBox(std::ifstream& settings) : blocks(std::vector<std::unique_ptr<Block> >()) {
    size_t in_dim;
    size_t out_dim;
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

inline Eigen::VectorXd BlackBox::evaluate(const Eigen::VectorXd& x) const {
    Eigen::VectorXd result = x;

    for (int i = 0; i < blocks_cnt; ++i) {
        result = blocks[i]->evaluate(result);
    }

    return result;
}

inline void BlackBox::tuning(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    std::vector<std::unique_ptr<Eigen::VectorXd> > remember_output;
    remember_output.emplace_back(std::make_unique<Eigen::VectorXd>(blocks[0]->evaluate(x)));

    for (int i = 1; i < blocks_cnt; ++i) {
        remember_output.emplace_back(
            std::make_unique<Eigen::VectorXd>(blocks[i]->evaluate(*remember_output[i - 1])));
    }

    Eigen::RowVectorXd u = loss.gradient(*remember_output[blocks_cnt - 1], y);
    Eigen::RowVectorXd u_next;

    for (int i = blocks_cnt - 1; i > 0; --i) {
        u_next = blocks[i]->propogateBack(*remember_output[i - 1], u);
        blocks[i]->gradientDescent(*remember_output[i - 1], u);
        u = u_next;
    }

    blocks[0]->gradientDescent(x, u);
}

}  // namespace bbx
