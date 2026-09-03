#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <bbx/core/types.hpp>

namespace bbx {

class Linear {
public:
    Linear(double alpha = 1.0, double beta = 0.0) : alpha_(alpha), beta_(beta) {}

    double evaluate(const double x) const
    {
        return alpha_ * x + beta_;
    }

    double derivative(const double x) const
    {
        return alpha_;
    }

private:
    double alpha_;
    double beta_;
};

class Sigmoid {
public:
    double evaluate(const double x) const
    {
        return 1 / (1 + exp(-x));
    }

    double derivative(const double x) const
    {
        double s = evaluate(x);
        return s * (1.0 - s);
    }
};

class HardSigmoid {
public:
    double evaluate(const double x) const
    {
        return x < -2.5 ? 0 : (x > 2.5 ? 1 : 0.2 * x + 0.5);
    }

    double derivative(const double x) const
    {
        return x < -2.5 ? 0 : (x > 2.5 ? 0 : 0.2);
    }
};

class ReLU {
public:
    double evaluate(const double x) const
    {
        return x >= 0 ? x : 0;
    }

    double derivative(const double x) const
    {
        return x >= 0 ? 1 : 0;
    }
};

class LReLU {
public:
    LReLU(double alpha = 0.03) : alpha_(alpha) {}

    double evaluate(const double x) const
    {
        return x >= 0 ? x : alpha_ * x;
    }

    double derivative(const double x) const
    {
        return x >= 0 ? 1 : alpha_;
    }

private:
    double alpha_;
};

class ELU {
public:
    ELU(double alpha = 1.0) : alpha_(alpha) {}

    double evaluate(const double x) const
    {
        return x >= 0 ? x : alpha_ * (std::exp(x) - 1);
    }

    double derivative(const double x) const
    {
        return x >= 0 ? 1 : alpha_ * std::exp(x);
    }

private:
    double alpha_;
};

class SELU {
public:
    SELU(double alpha = 1.67326324, double scale = 1.05070098) : alpha_(alpha), scale_(scale) {}

    double evaluate(const double x) const
    {
        return x >= 0 ? scale_ * x : scale_ * alpha_ * (std::exp(x) - 1);
    }

    double derivative(const double x) const
    {
        return x >= 0 ? scale_ : scale_ * alpha_ * std::exp(x);
    }

private:
    double alpha_;
    double scale_;
};

class GELU {
public:
    double evaluate(const double x) const
    {
        return 0.5 * x * (1 + std::tanh(0.7978845608 * (x + 0.044715 * std::pow(x, 3))));
    }

    double derivative(const double x) const
    {
        double x_cube = std::pow(x, 3);
        double g = 0.0356074 * x_cube + 0.797885 * x;
        double tmp = std::cosh(g);
        return 0.5 * std::tanh(g) + (0.053411 * x_cube + 0.398942 * x) / (tmp * tmp) + 0.5;
    }
};

class Exponential {
public:
    double evaluate(const double x) const
    {
        return std::exp(x);
    }

    double derivative(const double x) const
    {
        return std::exp(x);
    }
};

class Swish {
public:
    Swish(double beta = 1.0) : beta_(beta) {}

    double evaluate(const double x) const
    {
        return x / (1 + std::exp(-x * beta_));
    }

    double derivative(const double x) const
    {
        double sig = 1.0 / (1.0 + std::exp(-x * beta_));
        return sig * (1.0 + x * beta_ * (1.0 - sig));
    }

private:
    double beta_;
};

class Softplus {
public:
    double evaluate(const double x) const
    {
        return x >= 20 ? x : std::log(1 + std::exp(x));
    }

    double derivative(const double x) const
    {
        return 1 / (1 + std::exp(-x));
    }
};

class Softsign {
public:
    double evaluate(const double x) const
    {
        return x / (std::abs(x) + 1); 
    }

    double derivative(const double x) const
    {
        return 1 / std::pow(std::abs(x) + 1, 2);
    }
};

class Tanh {
public:
    double evaluate(const double x) const
    {
        return std::tanh(x);
    }

    double derivative(const double x) const
    {
        return 1 / std::pow(std::cosh(x), 2);
    }
};

}  // namespace bbx
