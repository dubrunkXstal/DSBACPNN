#pragma once

#include <bbx/types.hpp>
#include <cmath>
#include <functional>

namespace bbx {

using LRSchedule = std::function<double(int)>;

struct ConstantLR {
    double lr;

    explicit ConstantLR(double lr = 0.01) : lr(lr) {}

    double operator()(int) const { return lr; }
};

struct TimeDecayLR {
    double lambda;
    double s0;
    double p;

    explicit TimeDecayLR(double lambda = 1.0, double s0 = 1.0, double p = 0.5)
        : lambda(lambda), s0(s0), p(p) {}

    double operator()(int iteration) const { return lambda * std::pow(s0 / (s0 + iteration), p); }
};

}  // namespace bbx
