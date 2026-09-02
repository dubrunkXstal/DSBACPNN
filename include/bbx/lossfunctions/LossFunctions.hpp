#pragma once

#include <algorithm>
#include <cmath>
#include <bbx/types.hpp>

namespace bbx {

class L2NormSquared {
   public:
    double distance(const Vector& z, const Vector& y) const
    {
        return (z - y).squaredNorm();
    }

    RowVector gradient(const Vector& z, const Vector& y) const;
};

class AbsoluteError {
   public:
    double distance(const Vector& z, const Vector& y) const
    {
        return (z - y).cwiseAbs().sum();
    }

    RowVector gradient(const Vector& z, const Vector& y) const
    {
        return (z - y).cwiseSign();
    }
};

class HuberLoss {
public:
    HuberLoss(double delta = 1.0) : delta_(delta) {}

    double distance(const Vector& z, const Vector& y) const
    {
        double total = 0.0;
        for (int i = 0; i < z.rows(); ++i) {
            double d = z[i] - y[i];
            double ad = std::abs(d);
            total += ad <= delta_ ? 0.5 * d * d : delta_ * (ad - 0.5 * delta_);
        }
        return total;
    }

    RowVector gradient(const Vector& z, const Vector& y) const
    {
        RowVector result(z.rows());
        for (int i = 0; i < z.rows(); ++i) {
            double d = z[i] - y[i];
            result[i] = std::abs(d) <= delta_ ? d : delta_ * (d > 0 ? 1.0 : -1.0);
        }
        return result;
    }

private:
    double delta_;
};

class LogCosh {
public:
    double distance(const Vector& z, const Vector& y) const
    {
        constexpr double log2 = 0.6931471805599453;
        auto ax = (z - y).array().abs();
        return (ax + (-2.0 * ax).exp().log1p() - log2).sum();
    }

    RowVector gradient(const Vector& z, const Vector& y) const
    {
        return (z - y).array().tanh().matrix().transpose();
    }
};

class BinaryCrossEntropy {
public:
    double distance(const Vector& z, const Vector& y) const
    {
        constexpr double eps = 1e-15;
        double total = 0.0;
        for (int i = 0; i < z.rows(); ++i) {
            double zi = std::clamp(z[i], eps, 1.0 - eps);
            total += -y[i] * std::log(zi) - (1.0 - y[i]) * std::log(1.0 - zi);
        }
        return total;
    }

    RowVector gradient(const Vector& z, const Vector& y) const
    {
        constexpr double eps = 1e-15;
        RowVector result(z.rows());
        for (int i = 0; i < z.rows(); ++i) {
            double zi = std::clamp(z[i], eps, 1.0 - eps);
            result[i] = (zi - y[i]) / (zi * (1.0 - zi));
        }
        return result;
    }
};

class CategorialCrossEntropy {
public:
    double distance(const Vector& z, const Vector& y) const
    {
        constexpr double eps = 1e-15;
        double total = 0.0;
        for (int i = 0; i < z.rows(); ++i) {
            total += -y[i] * std::log(std::max(z[i], eps));
        }
        return total;
    }

    RowVector gradient(const Vector& z, const Vector& y) const
    {
        constexpr double eps = 1e-15;
        RowVector result(z.rows());
        for (int i = 0; i < z.rows(); ++i) {
            result[i] = -y[i] / std::max(z[i], eps);
        }
        return result;
    }
};

// Implementation

inline RowVector L2NormSquared::gradient(const Vector& z, const Vector& y) const
{
    RowVector result(z.rows());

    for (int i = 0; i < z.rows(); ++i) {
        result[i] = 2 * (z[i] - y[i]);
    }

    return result;
}

}  // namespace bbx
