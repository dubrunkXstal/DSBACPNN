#include <Eigen/Core>
#include <array>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <map>

// double fastSigmoidModified(double x) {
//     return x/(2 + 2 * abs(x)) + 0.5;
// }

// double fastSigmoidModifiedDerivative(double x) {
//     return x/(2 + 2 * abs(x)) + 0.5;
// }


// Это не быстрая сигмоида, а обычная, но мне лень переписывать названия функций
// double fastSigmoidModified(double x) {
//     return 1/(1 + exp(-x));
// }

// double fastSigmoidModifiedDerivative(double x) {
//     return exp(-x)/pow(1 + exp(-x), 2);
// }


// Это не сигмоиды но мне лень переписывать
double fastSigmoidModified(double x) {
    if (x < 0) { return 0; }

    return x;
}

double fastSigmoidModifiedDerivative(double x) {
    if (x < 0) { return 0; }

    return 1;
}


Eigen::MatrixXd makeDiagonal(const Eigen::VectorXd& v) {
    Eigen::MatrixXd m(v.rows(), v.rows());

    for (int i = 0; i < v.rows(); ++i) {
        m.col(i)[i] = v[i];
    }

    return m;
}


double gradient_descent_step = 0.01;


class Layer {
private:
    // Eigen::MatrixXd A;
    // Eigen::VectorXd b;
    int in_size;
    int out_size;
    // std::pair<Eigen::MatrixXd, Eigen::VectorXd> theta;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;

public:
    Layer(int in_size, int out_size) :
        in_size(in_size),
        out_size(out_size),
        // theta(std::pair<Eigen::MatrixXd, Eigen::VectorXd>(Eigen::MatrixXd::Random(out_size, in_size), Eigen::VectorXd::Random(out_size))) {}
        A(Eigen::MatrixXd::Random(out_size, in_size)),
        b(Eigen::VectorXd::Random(out_size)) {}

    Eigen::VectorXd linear(const Eigen::VectorXd& x) const {
        if (x.rows() != in_size) {
            // std::cout << x << "AAAAAAAA"<< '\n';
            // std::cout << A << '\n';
            throw std::runtime_error("Wrong dimension in input."); }

        return A * x + b;
    }

    Eigen::VectorXd nonLinear(const Eigen::VectorXd& x, double(*func_ptr)(double)) const {
        Eigen::VectorXd z = x;

        for (double& i : z) {
            i = func_ptr(i);
        }

        return z;
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const {
        return nonLinear(linear(x), &fastSigmoidModified);
    }

    Eigen::MatrixXd grad_A(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const {
        return makeDiagonal(nonLinear(linear(x), &fastSigmoidModifiedDerivative)) * u.transpose() * x.transpose();
    }

    Eigen::VectorXd grad_b(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const {
        return makeDiagonal(nonLinear(linear(x), &fastSigmoidModifiedDerivative)) * u.transpose();
    }

    Eigen::RowVectorXd grad_x(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const {
        return u * nonLinear(linear(x), &fastSigmoidModifiedDerivative) * A;
    }

    void gradientDescent(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) {
        A -= grad_A(x, u) * gradient_descent_step;
        b -= grad_b(x, u) * gradient_descent_step;
    }

    Eigen::RowVectorXd pullBackU(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) const {
        // std::cout << x;
        return grad_x(x, u);
    }

    std::pair<Eigen::MatrixXd, Eigen::VectorXd> getTheta() const {
        return std::pair<Eigen::MatrixXd, Eigen::VectorXd>(A, b);
    }
};

Eigen::VectorXd refFunction(Eigen::VectorXd& x) {
    double s = 0;

    for (int i = 0; i < 10; ++i) {
        s += pow(-x[i], i / 2);
    }

    Eigen::VectorXd v(1);

    v << s;

    return v;
}

double distance(const Eigen::VectorXd& z, const Eigen::VectorXd& y) {
    double dist = 0;

    for (int i = 0; i < y.rows(); ++i) {
        dist += pow(y[i] - z[i], 2);
    }

    return dist;
}

Eigen::RowVectorXd distanceRow(const Eigen::VectorXd& z, const Eigen::VectorXd& y) {
    Eigen::RowVectorXd v(z.rows());

    for (int i = 0; i < z.rows(); ++i) {
        v << 2 * (z[i] - y[i]);
    }

    return v;
}

Eigen::VectorXd NeuralNet(std::array<Layer, 2>& Net, Eigen::VectorXd& x) {
    return Net[1].evaluate(Net[0].evaluate(x));
}

double fine(std::array<Layer, 2>& Net, std::array<Eigen::VectorXd, 1000>& sample) {
    double sum = 0;

    for (int i = 0; i < 1000; ++i) {
        sum += distance(NeuralNet(Net, sample[i]), refFunction(sample[i]));
    }

    return sum / 1000;
}


int main() {
    Layer l1(7, 3);
    Eigen::VectorXd v(7);
    v << 1, 2, 3, 4, 5, 6, 7;



    std::array<Eigen::VectorXd, 1000> sample;

    for (Eigen::VectorXd& i : sample) {
        i = Eigen::VectorXd::Random(10) * 100;
    }

    std::array<Layer, 2> Net{Layer(10, 100), Layer(100, 1)};

    Eigen::VectorXd inp = Eigen::VectorXd::Random(10);

    while (!inp.isZero()) {
        for (int _ = 0; _ < 10; ++_) { std::cin >> inp[_]; }

        std::cout << "Ref: " << refFunction(inp) << '\n';
        std::cout << "NNet: " << NeuralNet(Net, inp) << '\n' << '\n';
    }

    for (int iterations = 0; iterations < 1; ++iterations) {
        std::cout << "----- Iteration " << iterations << " -----\n";
        Eigen::RowVectorXd sum_u(Eigen::RowVectorXd::Zero(1));

        for (Eigen::VectorXd& i : sample) {
            sum_u += distanceRow(NeuralNet(Net, i), refFunction(i));
        }

        Eigen::RowVectorXd average_u = sum_u / 1000;

        for (Eigen::VectorXd& i : sample) {
            Eigen::RowVectorXd u_next = Net[1].pullBackU(Net[0].evaluate(i), average_u);
            Net[1].gradientDescent(Net[0].evaluate(i), average_u);
            Net[0].gradientDescent(i, u_next);
        }
    }

    std::cout << "After train: \n";

    inp = Eigen::VectorXd::Random(10);
    while (!inp.isZero()) {
        for (int _ = 0; _ < 10; ++_) { std::cin >> inp[_]; }

        std::cout << "Ref: " << refFunction(inp) << '\n';
        std::cout << "NNet: " << NeuralNet(Net, inp) << '\n' << '\n';
    }

    // std::cout << l1.getTheta().first << '\n' << '\n' << l1.getTheta().second << '\n' << '\n' << l1.evaluate(v) << '\n';


    return 0;
}
