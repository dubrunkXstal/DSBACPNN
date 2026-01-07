#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sstream>
#include <random>
#include <stdexcept>
#include <cmath>


struct Sigmoid {
    double evaluate(const double x) const
    {
        if (x > 10) { return 1; }

        if (x < -10) { return 0; }

        return 1/(1 + exp(-x));
    }

    double derivative(const double x) const
    {
        if (x > 10) { return 0; }

        if (x < -10) { return 0; }

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
        Eigen::MatrixXd z(x.rows(), x.rows());

        for (int i = 0; i < x.rows(); ++i) {
            z.row(i)[i] = derivative(x[i]);
        }

        return z;
    }
};


double MODULO = 1299827;

struct Relu {
    double evaluate(const double x) const
    {
        if (x < 0) { return 0; }

        return std::fmod(x, MODULO);
    }

    double derivative(const double x) const
    {
        if (x < 0) { return 0; }

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


struct Compare {
    double evaluate(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const
    {
        return pow((z - y).norm(), 2);
    }

    Eigen::RowVectorXd derivative(const Eigen::VectorXd& z, const Eigen::VectorXd& y) const
    {
        Eigen::RowVectorXd result(z.rows());

        for (int i = 0; i < z.rows(); ++i) {
            result[i] = 2 * (y[i] - z[i]);
        }

        return result;
    }
};


double GRADIENT_STEP = 0.001;

class Block {
private:
    int in_dim;
    int out_dim;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Relu sigma;

public:
    Block(int in_dim, int out_dim, std::stringstream& theta) :
        in_dim(in_dim),
        out_dim(out_dim),
        A(Eigen::MatrixXd(out_dim, in_dim)),
        b(Eigen::VectorXd(out_dim))
    {
        for (int j = 0; j < in_dim; ++j) {
            for (int i = 0; i < out_dim; ++i) {
                theta >> A.col(j)[i];
            }
        }

        for (int i = 0; i < out_dim; ++i) {
            theta >> b[i];
        }
    }


    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const
    {
        return sigma.evaluate(A * x + b);
    }


    Eigen::MatrixXd grad_A(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
    {
        return sigma.derivative(A * x + b) * u.transpose() * x.transpose();
    }

    Eigen::VectorXd grad_b(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
    {
        return sigma.derivative(A * x + b) * u.transpose();
    }


    void gradientDescent(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u)
    {
        A -= grad_A(x, u) * GRADIENT_STEP;
        b -= grad_b(x, u) * GRADIENT_STEP;
    }


    Eigen::RowVectorXd propogateBack(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const
    {
        return u * sigma.derivative(A * x + b) * A;
    }
};


class BlackBox {
private:
    int blocks_cnt;
    std::vector<Block*> blocks;
    Compare compare;

public:
    BlackBox(int blocks_cnt, std::ifstream& file) :
        blocks_cnt(blocks_cnt),
        blocks(std::vector<Block*>(blocks_cnt))
    {
        int in_dim;
        int out_dim;
        std::string line;
        std::stringstream ss;

        for (int i = 0; i < blocks_cnt; ++i) {
            getline(file, line);
            ss = std::stringstream(line);
            ss >> in_dim >> out_dim;

            getline(file, line);
            ss = std::stringstream(line);
            blocks[i] = new Block(in_dim, out_dim, ss);
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

    Eigen::VectorXd evaluateAt(const Eigen::VectorXd& x, int at) const
    {
        if (at < 0 || at >= blocks_cnt) { throw std::runtime_error("There is no block at such index!"); }

        Eigen::VectorXd result = x;

        for (int i = 0; i <= at; ++i) {
            result = blocks[i]->evaluate(result);
        }

        return result;
    }

    void tuning(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
    {
        Eigen::RowVectorXd u = compare.derivative(evaluate(x), y);
        Eigen::RowVectorXd u_next;

        for (int i = blocks_cnt - 1; i > 0; --i) {
            u_next = blocks[i]->propogateBack(evaluateAt(x, i - 1), u);

            blocks[i]->gradientDescent(evaluateAt(x, i - 1), u);
            u = u_next;
        }

        blocks[0]->gradientDescent(x, u);
    }
};


Eigen::VectorXd refFunction(Eigen::VectorXd& x) {  // f(x, y) = (y, x)
    Eigen::VectorXd v(2);

    v << x[1], x[0];

    return v;
}

int main() {
    std::ifstream file("theta.txt");

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file!" << '\n';
        return 1;
    }

    BlackBox bb(2, file);

    const size_t sample_size = 1;
    std::array<Eigen::VectorXd, sample_size> sample;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.5, 1);

    for (Eigen::VectorXd& i : sample) {
        i = Eigen::VectorXd(2);
        // i[0] = distribution(generator);
        // i[1] = distribution(generator);
        i[0] = 3;
        i[1] = 4;
    }


    std::cout << "--- Before train: ---\n\n";
    Eigen::VectorXd input = Eigen::VectorXd(2);

    do {
        std::cout << "Input vector in RR^2: ";
        for (int i = 0; i < 2; ++i) { std::cin >> input[i]; }

        std::cout << "\nRef:\n" << refFunction(input) << '\n';
        std::cout << "BBox:\n" << bb.evaluate(input) << '\n' << '\n';
    } while (!input.isZero());


    for (int iterations = 0; iterations < 100000; ++iterations) {
        // std::cout << "- Iteration " << iterations << " -\n";
        for (Eigen::VectorXd& i : sample) {
            bb.tuning(i, refFunction(i));
        }
    }

    std::cout << "--- After train: ---\n\n";
    input = Eigen::VectorXd(2);

    do {
        std::cout << "Input vector in RR^2: ";
        for (int i = 0; i < 2; ++i) { std::cin >> input[i]; }

        std::cout << "\nRef:\n" << refFunction(input) << '\n';
        std::cout << "BBox:\n" << bb.evaluate(input) << '\n' << '\n';
    } while (!input.isZero());


    return 0;
}
