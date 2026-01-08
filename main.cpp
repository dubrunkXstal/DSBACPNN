#include <cstdio>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <string>
#include "EigenRand/EigenRand/EigenRand"
#include "AnyActivationFunction.h"


Eigen::Rand::P8_mt19937_64 urng{42};

double GRADIENT_STEP = 0.01;


struct Sigmoid {
    Sigmoid() = default;

    Sigmoid(const Sigmoid& other) = delete;

    Sigmoid(Sigmoid&& other) = default;

    Sigmoid& operator=(const Sigmoid& other) = delete;

    Sigmoid& operator=(Sigmoid&& other) = default;

    ~Sigmoid() = default;


    double evaluate(const double x) const
    {
        if (isnan(x)) { return 0; }

        if (x > 14) { return 1; }

        if (x < -20) { return 0; }

        return 1/(1 + exp(-x));
    }

    double derivative(const double x) const
    {
        if (isnan(x)) { return 0; }

        if (x > 14) { return 0; }

        if (x < -20) { return 0; }

        return exp(-x)/pow(1 + exp(-x), 2);
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd z = x;

        for (double& i : z) { i = evaluate(i); }

        return z;
    }

    Eigen::MatrixXd derivative(const Eigen::VectorXd& x) const
    {
        Eigen::MatrixXd z = Eigen::MatrixXd::Zero(x.rows(), x.rows());

        for (int i = 0; i < x.rows(); ++i) {
            z.row(i)[i] = derivative(x[i]);
        }

        return z;
    }
};


struct Relu {
    Relu() = default;

    Relu(const Relu& other) = delete;

    Relu(Relu&& other) = default;

    Relu& operator=(const Relu& other) = delete;

    Relu& operator=(Relu&& other) = default;

    ~Relu() = default;


    double evaluate(const double x) const
    {
        if (x < 0 || isnan(x)) { return 0; }

        return x;
    }

    double derivative(const double x) const
    {
        if (x < 0 || isnan(x)) { return 0; }

        return 1;
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd z = x;

        for (double& i : z) { i = evaluate(i); }

        return z;
    }

    Eigen::MatrixXd derivative(const Eigen::VectorXd& x) const
    {
        Eigen::MatrixXd z(x.rows(), x.rows());

        for (int i = 0; i < x.rows(); ++i) {
            z.row(i)[i] = derivative(x[i]);
        }

        return z;
    }
};


struct LossFunction {
    double distance(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const
    {
        return pow((z - y).norm(), 2);
    }

    Eigen::RowVectorXd gradient(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const
    {
        Eigen::RowVectorXd result(z.rows());

        for (int i = 0; i < z.rows(); ++i) {
            result[i] = 2 * (z[i] - y[i]);
        }

        return result;
    }
};


class Block {
private:
    size_t in_dim;
    size_t out_dim;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    CAny sigma;

public:
    std::unique_ptr<Eigen::VectorXd> remember_output;

    Block(size_t in_dim, size_t out_dim, CAny&& sigma) :
        in_dim(in_dim),
        out_dim(out_dim),
        A(Eigen::Rand::normal<Eigen::MatrixXd>(out_dim, in_dim, urng)),
        b(Eigen::VectorXd(Eigen::Rand::normal<Eigen::MatrixXd>(out_dim, 1, urng))),
        sigma(std::move(sigma)),
        remember_output(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd(out_dim))) {}


    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd w = sigma->evaluate((A * x + b).eval());
        *remember_output = w;

        return w;
    }


    Eigen::MatrixXd grad_A(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
    {
        return (sigma->derivative((A * x + b).eval()) * u.transpose() * x.transpose()).eval();
    }

    Eigen::VectorXd grad_b(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
    {
        return (sigma->derivative((A * x + b).eval()) * u.transpose()).eval();
    }

    void gradientDescent(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u)
    {
        A = (A - grad_A(x, u) * GRADIENT_STEP).eval();
        b = (b - grad_b(x, u) * GRADIENT_STEP).eval();
    }


    Eigen::RowVectorXd propogateBack(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
    {
        return (u * sigma->derivative((A * x + b).eval()) * A).eval();
    }
};


class BlackBox {
private:
    size_t blocks_cnt;
    std::vector<std::unique_ptr<Block> > blocks;
    LossFunction loss;

public:
    BlackBox(size_t blocks_cnt, std::ifstream& file) :
        blocks_cnt(blocks_cnt),
        blocks(std::vector<std::unique_ptr<Block> >(blocks_cnt))
    {
        size_t in_dim;
        size_t out_dim;
        std::string activaton;
        std::string line;
        std::stringstream ss;

        for (int i = 0; i < blocks_cnt; ++i) {
            getline(file, line);
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


    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd result = x;

        for (int i = 0; i < blocks_cnt; ++i) {
            result = blocks[i]->evaluate(result);
        }

        return result;
    }


    void tuning(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
    {
        Eigen::RowVectorXd u = loss.gradient(*blocks[blocks_cnt - 1]->remember_output, y);
        Eigen::RowVectorXd u_next;

        for (int i = blocks_cnt - 1; i > 0; --i) {
            u_next = blocks[i]->propogateBack(*blocks[i - 1]->remember_output, u);
            blocks[i]->gradientDescent(*blocks[i - 1]->remember_output, u);
            u = u_next;
        }

        blocks[0]->gradientDescent(x, u);
    }
};


int main() {
    std::ifstream settings("settings.txt");
    if (!settings.is_open()) {
        std::cerr << "Error: Unable to open settings.txt!" << '\n';
        return 1;
    }

    std::string line;
    std::getline(settings, line);
    std::stringstream ss(line);


    size_t blocks_cnt;
    ss >> blocks_cnt;
    BlackBox bb(blocks_cnt, settings);

    size_t epochs_cnt = 1;
    size_t sample_size = 5000;  // Сам MNIST по размеру - 60000.

    for (int e = 1; e <= epochs_cnt; ++e) {
        std::ifstream mnist_sample("MNIST/mnist_train.csv");
        if (!mnist_sample.is_open()) {
            std::cerr << "Error: Unable to open mnist_train.txt!" << '\n';
            return 1;
        }

        std::string line;
        std::stringstream ss;
        std::string integer;
        int i = 0;

        while (mnist_sample.good() && std::getline(mnist_sample, line) && i < sample_size) {
            std::cout << "epoch " << e << "/" << epochs_cnt << " : - " << ++i << " -\n";

            ss = std::stringstream(line);
            std::getline(ss, integer, ',');

            Eigen::VectorXd y = Eigen::VectorXd::Zero(10);
            y[std::stoi(integer)] = 1;

            Eigen::VectorXd x(784);
            for (double& i : x) {
                std::getline(ss, integer, ',');
                i = std::stoi(integer) / 255.0;
            }

            bb.evaluate(x);
            bb.tuning(x, y);
        }

        mnist_sample.close();
    }


    std::ifstream mnist_test("MNIST/mnist_test.csv");
    if (!mnist_test.is_open()) {
        std::cerr << "Error: Unable to open mnist_test.txt!" << '\n';
        return 1;
    }

    std::string integer;

    while (mnist_test.good() && std::getline(mnist_test, line)) {
        std::string go;
        std::cin >> go;

        ss = std::stringstream(line);
        std::getline(ss, integer, ',');

        Eigen::VectorXd y = Eigen::VectorXd::Zero(10);
        y[std::stoi(integer)] = 1;
        std::cout << "Reference: " << integer << '\n';

        Eigen::VectorXd x(784);
        for (double& i : x) {
            std::getline(ss, integer, ',');
            i = std::stoi(integer) / 255.0;
        }

         Eigen::VectorXd res = bb.evaluate(x);

         int ind_max = 0;
         for (int i = 1; i < 10; ++i) {
             if (res[i] > res[ind_max]) { ind_max = i; }
         }

         std::cout << "Prediction: " << ind_max << '\n';
    }

    return 0;
}
