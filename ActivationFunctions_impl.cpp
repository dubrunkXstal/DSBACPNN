#ifndef ANYMOVABLE_IMPL_CPP
#define ANYMOVABLE_IMPL_CPP

#include "ActivationFunctions.h"


Sigmoid::Sigmoid() = default;

Sigmoid::Sigmoid(Sigmoid&& other) noexcept = default;

Sigmoid& Sigmoid::operator=(Sigmoid&& other) noexcept = default;

Sigmoid::~Sigmoid() = default;


double Sigmoid::evaluate(const double x) const
{
    if (isnan(x)) { return 0; }

    if (x > 14) { return 1; }

    if (x < -20) { return 0; }

    return 1/(1 + exp(-x));
}

double Sigmoid::derivative(const double x) const
{
    if (isnan(x)) { return 0; }

    if (x > 14) { return 0; }

    if (x < -20) { return 0; }

    return exp(-x)/pow(1 + exp(-x), 2);
}

Eigen::VectorXd Sigmoid::evaluate(const Eigen::VectorXd& x) const
{
    Eigen::VectorXd z = x;

    for (double& i : z) { i = evaluate(i); }

    return z;
}

Eigen::VectorXd Sigmoid::derivative(const Eigen::VectorXd& x) const
{
    Eigen::VectorXd z = x;

    for (double& i : z) { i = derivative(i); }

    return z;
}


Relu::Relu() = default;

Relu::Relu(Relu&& other) noexcept = default;

Relu& Relu::operator=(Relu&& other) noexcept = default;

Relu::~Relu() = default;


double Relu::evaluate(const double x) const
{
    if (x < 0 || isnan(x)) { return 0; }

    return x;
}

double Relu::derivative(const double x) const
{
    if (x < 0 || isnan(x)) { return 0; }

    return 1;
}

Eigen::VectorXd Relu::evaluate(const Eigen::VectorXd& x) const
{
    Eigen::VectorXd z = x;

    for (double& i : z) { i = evaluate(i); }

    return z;
}

Eigen::VectorXd Relu::derivative(const Eigen::VectorXd& x) const
{
    Eigen::VectorXd z = x;

    for (double& i : z) { i = derivative(i); }

    return z;
}

#endif  // ANYMOVABLE_IMPL_CPP
