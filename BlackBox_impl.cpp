#ifndef BLACKBOX_IMPL_CPP
#define BLACKBOX_IMPL_CPP

#include "BlackBox.h"


Eigen::Rand::P8_mt19937_64 urng{42};

double GRADIENT_STEP = 0.01;


double LossFunction::distance(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const
{
    return pow((z - y).norm(), 2);
}

Eigen::RowVectorXd LossFunction::gradient(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const
{
    Eigen::RowVectorXd result(z.rows());

    for (int i = 0; i < z.rows(); ++i) {
        result[i] = 2 * (z[i] - y[i]);
    }

    return result;
}


Block::Block(size_t in_dim, size_t out_dim, CAny&& sigma) :
    in_dim(in_dim),
    out_dim(out_dim),
    A(Eigen::Rand::normal<Eigen::MatrixXd>(out_dim, in_dim, urng)),
    b(Eigen::VectorXd(Eigen::Rand::normal<Eigen::MatrixXd>(out_dim, 1, urng))),
    sigma(std::move(sigma)) {}

Eigen::VectorXd Block::evaluate(const Eigen::VectorXd& x) const
{
    return sigma->evaluate(A * x + b);
}

Eigen::MatrixXd Block::grad_A(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
{
    return sigma->derivative(A * x + b).asDiagonal() * u.transpose() * x.transpose();
}

Eigen::VectorXd Block::grad_b(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
{
    return sigma->derivative(A * x + b).asDiagonal() * u.transpose();
}

void Block::gradientDescent(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u)
{
    A -= grad_A(x, u) * GRADIENT_STEP;
    b -= grad_b(x, u) * GRADIENT_STEP;
}

// void Block::gradientDescent(const Eigen::MatrixXd& x_batch, const Eigen::MatrixXd& u_batch, size_t batch_size)
// {
//     Eigen::MatrixXd sum_grad_A = Eigen::MatrixXd::Zero(u_batch.cols(), x_batch.rows());
//     Eigen::VectorXd sum_grad_b = Eigen::VectorXd::Zero(u_batch.cols());

//     for (int i = 0; i < batch_size; ++i) {
//         sum_grad_A += grad_A(x_batch.col(i), u_batch.row(i));
//         sum_grad_b += grad_b(x_batch.col(i), u_batch.row(i));
//     }

//     A -= sum_grad_A / batch_size * GRADIENT_STEP;
//     b -= sum_grad_b / batch_size * GRADIENT_STEP;
// }

Eigen::RowVectorXd Block::propogateBack(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
{
    return u * sigma->derivative(A * x + b).asDiagonal() * A;
}


BlackBox::BlackBox(size_t blocks_cnt, std::ifstream& settings) :
    blocks_cnt(blocks_cnt),
    blocks(std::vector<std::unique_ptr<Block> >(blocks_cnt))
{
    size_t in_dim;
    size_t out_dim;
    std::string activaton;
    std::string line;
    std::stringstream ss;

    for (int i = 0; i < blocks_cnt; ++i) {
        getline(settings, line);
        ss = std::stringstream(line);
        ss >> in_dim >> out_dim >> activaton;

        if (activaton == "sigmoid") {
            blocks[i] = std::make_unique<Block>(in_dim, out_dim, Sigmoid());
        }
        else if (activaton == "relu") {
            blocks[i] = std::make_unique<Block>(in_dim, out_dim, Relu());
        }
        else {
            throw std::runtime_error("Didn't found activaton function for the block.");
        }
    }
}

Eigen::VectorXd BlackBox::evaluate(const Eigen::VectorXd& x) const
{
    Eigen::VectorXd result = x;

    for (int i = 0; i < blocks_cnt; ++i) {
        result = blocks[i]->evaluate(result);
    }

    return result;
}

void BlackBox::tuning(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
{
    std::vector<std::unique_ptr<Eigen::VectorXd> > remember_output;
    remember_output.emplace_back(std::make_unique<Eigen::VectorXd>(blocks[0]->evaluate(x)));

    for (int i = 1; i < blocks_cnt; ++i) {
        remember_output.emplace_back(std::make_unique<Eigen::VectorXd>(blocks[i]->evaluate(*remember_output[i - 1])));
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


// void BlackBox::tuning(const Eigen::MatrixXd& x_batch, const Eigen::MatrixXd& y_batch, size_t batch_size)
// {
//     std::vector<std::unique_ptr<Eigen::MatrixXd> > remember_output;  // dim x batch_size
//     remember_output.emplace_back(std::make_unique<Eigen::MatrixXd>(blocks[0]->evaluate(x_batch)));

//     for (int i = 1; i < blocks_cnt; ++i) {
//         remember_output.emplace_back(std::make_unique<Eigen::MatrixXd>(blocks[i]->evaluate(*remember_output[i - 1])));
//     }

//     Eigen::MatrixXd u = loss.gradient(*remember_output[blocks_cnt - 1], y_batch);  // batch_size x dim
//     Eigen::MatrixXd u_next;
//     Eigen::RowVectorXd average = Eigen::RowVectorXd::Zero(x_batch.rows());

//     for (int i = 0; i < batch_size; ++i) {
//         average += u.row(i);
//     }

//     average /= batch_size;


// }

#endif  // BLACKBOX_IMPL_CPP
