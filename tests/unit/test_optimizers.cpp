#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bbx/bbx.hpp>

using namespace bbx;
using Catch::Matchers::WithinAbs;

// ConstantLR

TEST_CASE("ConstantLR: возвращает одинаковое значение на любой итерации", "[schedule]")
{
    ConstantLR lr{0.01};
    REQUIRE_THAT(lr(0),   WithinAbs(0.01, 1e-12));
    REQUIRE_THAT(lr(1),   WithinAbs(0.01, 1e-12));
    REQUIRE_THAT(lr(999), WithinAbs(0.01, 1e-12));
}

// TimeDecayLR

TEST_CASE("TimeDecayLR: lr(t) = lambda * (s0 / (s0+t))^p", "[schedule]")
{
    // lambda=1, s0=1, p=1, значит lr(t) = 1/(1+t)
    TimeDecayLR lr{1.0, 1.0, 1.0};
    REQUIRE_THAT(lr(0), WithinAbs(1.0,  1e-10));
    REQUIRE_THAT(lr(1), WithinAbs(0.5,  1e-10));
    REQUIRE_THAT(lr(3), WithinAbs(0.25, 1e-10));
}

TEST_CASE("TimeDecayLR: строго убывает", "[schedule]")
{
    TimeDecayLR lr{1.0, 1.0, 0.5};
    REQUIRE(lr(0) > lr(1));
    REQUIRE(lr(1) > lr(10));
    REQUIRE(lr(10) > lr(100));
}

// VanillaDescent

TEST_CASE("VanillaDescent: первое обновление равно -lr * gradient", "[optimizer][vanilla]")
{
    VanillaDescent opt{ConstantLR{0.1}};
    Matrix grad = Matrix::Ones(2, 3);
    Matrix update = opt.computeUpdate(grad);
    REQUIRE(update.isApprox(-0.1 * Matrix::Ones(2, 3)));
}

TEST_CASE("VanillaDescent: направление обновления противоположно градиенту", "[optimizer][vanilla]")
{
    VanillaDescent opt{ConstantLR{0.01}};
    Matrix grad(2, 2);
    grad << 1.0, -2.0,
            3.0, -4.0;
    Matrix update = opt.computeUpdate(grad);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            REQUIRE(update(i, j) * grad(i, j) <= 0.0);
}

TEST_CASE("VanillaDescent: величина шага убывает с TimeDecayLR", "[optimizer][vanilla]")
{
    VanillaDescent opt{TimeDecayLR{1.0, 1.0, 1.0}};
    Matrix grad = Matrix::Ones(1, 1);
    double step1 = std::abs(opt.computeUpdate(grad)(0, 0));  // lr = 1/(1+0) = 1
    double step2 = std::abs(opt.computeUpdate(grad)(0, 0));  // lr = 1/(1+1) = 0.5
    REQUIRE(step2 < step1);
}

TEST_CASE("VanillaDescent: нулевой градиент дает нулевое обновление", "[optimizer][vanilla]")
{
    VanillaDescent opt;
    Matrix update = opt.computeUpdate(Matrix::Zero(3, 3));
    REQUIRE(update.isZero(1e-12));
}

// MomentumDescent

TEST_CASE("MomentumDescent: первый шаг равен -lr * gradient", "[optimizer][momentum]")
{
    MomentumDescent opt{ConstantLR{0.1}, 0.9};
    Matrix grad = Matrix::Ones(2, 2);
    // velocity_0 = 0; velocity_1 = beta*0 + lr*grad = 0.1*I
    Matrix update = opt.computeUpdate(grad);
    REQUIRE(update.isApprox(-0.1 * Matrix::Ones(2, 2)));
}

TEST_CASE("MomentumDescent: последующие шаги накапливают импульс", "[optimizer][momentum]")
{
    MomentumDescent opt{ConstantLR{0.1}, 0.9};
    Matrix grad = Matrix::Ones(1, 1);
    Matrix step1 = opt.computeUpdate(grad);
    Matrix step2 = opt.computeUpdate(grad);
    // step2 больше по модулю из-за накопления импульса
    REQUIRE(std::abs(step2(0, 0)) > std::abs(step1(0, 0)));
}

TEST_CASE("MomentumDescent: направление обновления противоположно градиенту", "[optimizer][momentum]")
{
    MomentumDescent opt{ConstantLR{0.01}, 0.9};
    Matrix grad(2, 2);
    grad << 2.0, -1.0, 3.0, -4.0;
    Matrix update = opt.computeUpdate(grad);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            REQUIRE(update(i, j) * grad(i, j) <= 0.0);
}

// Adam

TEST_CASE("Adam: направление обновления противоположно градиенту", "[optimizer][adam]")
{
    Adam opt{ConstantLR{0.001}};
    Matrix grad(2, 2);
    grad << 1.0, -2.0, 3.0, -4.0;
    Matrix update = opt.computeUpdate(grad);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            REQUIRE(update(i, j) * grad(i, j) < 0.0);
}

TEST_CASE("Adam: величина первого шага примерно равна lr", "[optimizer][adam]")
{
    Adam opt{ConstantLR{0.001}};
    // Для единичного градиента: m_hat = v_hat = 1, значит update примерно -lr
    Matrix grad = Matrix::Ones(3, 3);
    Matrix update = opt.computeUpdate(grad);
    REQUIRE_THAT(std::abs(update(0, 0)), WithinAbs(0.001, 1e-4));
}

TEST_CASE("Adam: нулевой градиент дает нулевое обновление", "[optimizer][adam]")
{
    Adam opt;
    Matrix update = opt.computeUpdate(Matrix::Zero(3, 3));
    REQUIRE(update.isZero(1e-12));
}

TEST_CASE("Adam: обновления конечны при одинаковых градиентах", "[optimizer][adam]")
{
    Adam opt{ConstantLR{0.001}};
    Matrix grad = Matrix::Ones(2, 2);
    for (int i = 0; i < 10; ++i) {
        Matrix update = opt.computeUpdate(grad);
        REQUIRE(update.array().isFinite().all());
    }
}
