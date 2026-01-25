#ifndef BLACKBOX_H
#define BLACKBOX_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include "EigenRand/EigenRand/EigenRand"
#include <memory>
#include <cstdio>
#include <fstream>
#include "ActivationFunctions.h"


class LossFunction {
public:
    double distance(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const;

    Eigen::RowVectorXd gradient(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const;
};


class Block {
    size_t in_dim;
    size_t out_dim;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    CAny sigma;

public:
    Block(size_t in_dim, size_t out_dim, CAny&& sigma);


    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const;

    Eigen::MatrixXd grad_A(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const;

    Eigen::VectorXd grad_b(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const;

    void gradientDescent(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u);

    Eigen::RowVectorXd propogateBack(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const;
};


class BlackBox {
    size_t blocks_cnt;
    std::vector<std::unique_ptr<Block> > blocks;
    LossFunction loss;

public:
    BlackBox(size_t blocks_cnt, std::ifstream& settings);


    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const;

    void tuning(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

    void tuning(const Eigen::MatrixXd& x_batch, const Eigen::MatrixXd& y_batch, size_t batch_size);
};

#include "BlackBox_impl.cpp"

#endif  // BLACKBOX_H
