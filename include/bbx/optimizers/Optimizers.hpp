#pragma once

#include <bbx/schedules/LearningRates.hpp>
#include <bbx/core/types.hpp>
#include <cmath>

namespace bbx {

class VanillaDescent {
   public:
    explicit VanillaDescent(LRSchedule lr_schedule = ConstantLR{0.01}) : lr_schedule_(std::move(lr_schedule))
    {}

    Matrix computeUpdate(const Matrix& gradient)
    {
        Matrix delta = -lr_schedule_(iteration_) * gradient;
        ++iteration_;
        return delta;
    }

   private:
    LRSchedule lr_schedule_;
    int iteration_ = 0;
};

class MomentumDescent {
   public:
    explicit MomentumDescent(LRSchedule lr_schedule = ConstantLR{0.01}, double beta = 0.9)
        : lr_schedule_(std::move(lr_schedule)), beta_(beta)
    {}

    Matrix computeUpdate(const Matrix& gradient);

   private:
    LRSchedule lr_schedule_;
    double beta_;
    Matrix velocity_;
    int iteration_ = 0;
};

class Adam {
   public:
    explicit Adam(LRSchedule lr_schedule = ConstantLR{0.001}, double beta1 = 0.9, double beta2 = 0.999,
                  double eps = 1e-8)
        : lr_schedule_(std::move(lr_schedule)), beta1_(beta1), beta2_(beta2), eps_(eps)
    {}

    Matrix computeUpdate(const Matrix& gradient);

   private:
    LRSchedule lr_schedule_;
    double beta1_;
    double beta2_;
    double eps_;
    Matrix m_;
    Matrix v_;
    int t_ = 0;
};

// Implementation

inline Matrix MomentumDescent::computeUpdate(const Matrix& gradient)
{
    if (velocity_.size() == 0) {
        velocity_ = Matrix::Zero(gradient.rows(), gradient.cols());
    }

    velocity_ = beta_ * velocity_ + lr_schedule_(iteration_) * gradient;
    ++iteration_;
    return -velocity_;
}

inline Matrix Adam::computeUpdate(const Matrix& gradient)
{
    if (m_.size() == 0) {
        m_ = Matrix::Zero(gradient.rows(), gradient.cols());
        v_ = Matrix::Zero(gradient.rows(), gradient.cols());
    }

    ++t_;
    m_ = beta1_ * m_ + (1.0 - beta1_) * gradient;
    v_ = beta2_ * v_ + (1.0 - beta2_) * gradient.array().square().matrix();

    Matrix m_est = m_ / (1.0 - std::pow(beta1_, t_));
    Matrix v_est = v_ / (1.0 - std::pow(beta2_, t_));

    return -lr_schedule_(t_ - 1) * (m_est.array() / (v_est.array().sqrt() + eps_)).matrix();
}

}  // namespace bbx
