#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bbx/bbx.hpp>
#include <cmath>

using namespace bbx;
using Catch::Matchers::WithinAbs;

// Численный градиент потерь по z[i] (центральная разность)
template <typename Loss>
static double num_grad(const Loss& loss, const Vector& z, const Vector& y, int i)
{
    constexpr double h = 1e-5;
    Vector zp = z, zm = z;
    zp[i] += h;
    zm[i] -= h;
    return (loss.distance(zp, y) - loss.distance(zm, y)) / (2.0 * h);
}

// Проверяет совпадение аналитического и численного градиента по всем компонентам
template <typename Loss>
static void check_gradient(const Loss& loss, const Vector& z, const Vector& y, double tol = 1e-5)
{
    RowVector grad = loss.gradient(z, y);
    for (int i = 0; i < (int)z.size(); ++i) {
        REQUIRE_THAT(grad[i], WithinAbs(num_grad(loss, z, y, i), tol));
    }
}

// L2NormSquared

TEST_CASE("L2NormSquared: расстояние для известных значений", "[loss][l2]")
{
    L2NormSquared loss;
    Vector z(3), y(3);
    z << 1.0, 2.0, 3.0;
    y << 1.0, 1.0, 1.0;
    // (0)^2 + (1)^2 + (2)^2 = 5
    REQUIRE_THAT(loss.distance(z, y), WithinAbs(5.0, 1e-10));
}

TEST_CASE("L2NormSquared: расстояние равно нулю для одинаковых векторов", "[loss][l2]")
{
    L2NormSquared loss;
    Vector v(4); v << 1.0, -2.0, 0.0, 5.0;
    REQUIRE_THAT(loss.distance(v, v), WithinAbs(0.0, 1e-12));
}

TEST_CASE("L2NormSquared: градиент равен 2*(z-y)", "[loss][l2]")
{
    L2NormSquared loss;
    Vector z(2), y(2);
    z << 3.0, 1.0;
    y << 1.0, 1.0;
    RowVector grad = loss.gradient(z, y);
    REQUIRE_THAT(grad[0], WithinAbs(4.0, 1e-10));
    REQUIRE_THAT(grad[1], WithinAbs(0.0, 1e-10));
}

TEST_CASE("L2NormSquared: градиент совпадает с численным", "[loss][l2]")
{
    L2NormSquared loss;
    Vector z(3), y(3);
    z << 2.0, -1.0, 0.5;
    y << 1.0,  1.0, 0.0;
    check_gradient(loss, z, y);
}

// AbsoluteError

TEST_CASE("AbsoluteError: расстояние", "[loss][mae]")
{
    AbsoluteError loss;
    Vector z(3), y(3);
    z << 1.0, -1.0, 2.0;
    y << 0.0,  0.0, 0.0;
    REQUIRE_THAT(loss.distance(z, y), WithinAbs(4.0, 1e-10));
}

TEST_CASE("AbsoluteError: градиент равен знаку разности", "[loss][mae]")
{
    AbsoluteError loss;
    Vector z(3), y(3);
    z << 2.0, -1.0, 0.5;
    y << 0.0,  0.0, 0.0;
    RowVector grad = loss.gradient(z, y);
    REQUIRE_THAT(grad[0], WithinAbs( 1.0, 1e-12));
    REQUIRE_THAT(grad[1], WithinAbs(-1.0, 1e-12));
    REQUIRE_THAT(grad[2], WithinAbs( 1.0, 1e-12));
}

TEST_CASE("AbsoluteError: градиент совпадает с численным (вдали от нуля)", "[loss][mae]")
{
    AbsoluteError loss;
    Vector z(3), y(3);
    z << 2.0, -1.5, 0.8;
    y << 0.0,  0.0, 0.0;
    check_gradient(loss, z, y);
}

// HuberLoss

TEST_CASE("HuberLoss: квадратичная область при |d| <= delta", "[loss][huber]")
{
    HuberLoss loss{1.0};
    Vector z(1), y(1);
    z << 0.5; y << 0.0;
    // 0.5 * 0.5^2 = 0.125
    REQUIRE_THAT(loss.distance(z, y), WithinAbs(0.125, 1e-10));
}

TEST_CASE("HuberLoss: линейная область при |d| > delta", "[loss][huber]")
{
    HuberLoss loss{1.0};
    Vector z(1), y(1);
    z << 2.0; y << 0.0;
    // delta*(|d| - 0.5*delta) = 1*(2 - 0.5) = 1.5
    REQUIRE_THAT(loss.distance(z, y), WithinAbs(1.5, 1e-10));
}

TEST_CASE("HuberLoss: непрерывность на границе delta", "[loss][huber]")
{
    HuberLoss loss{1.0};
    Vector z_in(1), z_out(1), y(1);
    z_in  << 1.0 - 1e-8;
    z_out << 1.0 + 1e-8;
    y << 0.0;
    REQUIRE_THAT(loss.distance(z_in, y), WithinAbs(loss.distance(z_out, y), 1e-6));
}

TEST_CASE("HuberLoss: расстояние равно нулю для одинаковых векторов", "[loss][huber]")
{
    HuberLoss loss{1.0};
    Vector v(3); v << 1.0, -2.0, 3.0;
    REQUIRE_THAT(loss.distance(v, v), WithinAbs(0.0, 1e-12));
}

TEST_CASE("HuberLoss: градиент совпадает с численным", "[loss][huber]")
{
    HuberLoss loss{1.0};
    Vector z(3), y(3);
    z << 0.5, 2.0, -0.3;   // первый в квадратичной зоне, второй в линейной
    y << 0.0, 0.0,  0.0;
    check_gradient(loss, z, y);
}

// LogCosh

TEST_CASE("LogCosh: расстояние равно нулю для одинаковых векторов", "[loss][logcosh]")
{
    LogCosh loss;
    Vector v(3); v << 1.0, -2.0, 0.5;
    REQUIRE_THAT(loss.distance(v, v), WithinAbs(0.0, 1e-12));
}

TEST_CASE("LogCosh: расстояние для известного значения", "[loss][logcosh]")
{
    LogCosh loss;
    Vector z(1), y(1);
    z << 1.0; y << 0.0;
    REQUIRE_THAT(loss.distance(z, y), WithinAbs(std::log(std::cosh(1.0)), 1e-8));
}

TEST_CASE("LogCosh: численная устойчивость для больших значений", "[loss][logcosh]")
{
    LogCosh loss;
    Vector z(1), y(1);
    z << 1000.0; y << 0.0;
    double d = loss.distance(z, y);
    REQUIRE(std::isfinite(d));
    // log(cosh(1000)) = 1000 + log1p(exp(-2000)) - log(2) ~ 999.307
    REQUIRE_THAT(d, WithinAbs(1000.0 - std::log(2.0), 1e-3));
}

TEST_CASE("LogCosh: градиент равен tanh разности", "[loss][logcosh]")
{
    LogCosh loss;
    Vector z(1), y(1);
    z << 1.0; y << 0.0;
    RowVector grad = loss.gradient(z, y);
    REQUIRE_THAT(grad[0], WithinAbs(std::tanh(1.0), 1e-10));
}

TEST_CASE("LogCosh: градиент совпадает с численным", "[loss][logcosh]")
{
    LogCosh loss;
    Vector z(3), y(3);
    z << 1.0, -0.5, 2.0;
    y << 0.0,  0.0, 0.0;
    check_gradient(loss, z, y);
}

// BinaryCrossEntropy

TEST_CASE("BinaryCrossEntropy: расстояние при z=0.5, y=1 равно ln(2)", "[loss][bce]")
{
    BinaryCrossEntropy loss;
    Vector z(1), y(1);
    z << 0.5; y << 1.0;
    REQUIRE_THAT(loss.distance(z, y), WithinAbs(std::log(2.0), 1e-8));
}

TEST_CASE("BinaryCrossEntropy: расстояние неотрицательно", "[loss][bce]")
{
    BinaryCrossEntropy loss;
    Vector z(3), y(3);
    z << 0.3, 0.7, 0.5;
    y << 1.0, 0.0, 1.0;
    REQUIRE(loss.distance(z, y) >= 0.0);
}

TEST_CASE("BinaryCrossEntropy: устойчивость для крайних z", "[loss][bce]")
{
    BinaryCrossEntropy loss;
    Vector z(2), y(2);
    z << 0.0, 1.0;
    y << 0.0, 1.0;
    REQUIRE(std::isfinite(loss.distance(z, y)));
}

TEST_CASE("BinaryCrossEntropy: градиент совпадает с численным", "[loss][bce]")
{
    BinaryCrossEntropy loss;
    Vector z(3), y(3);
    z << 0.3, 0.7, 0.5;
    y << 1.0, 0.0, 1.0;
    check_gradient(loss, z, y, 1e-4);
}

// CategorialCrossEntropy

TEST_CASE("CategorialCrossEntropy: расстояние для one-hot цели", "[loss][cce]")
{
    CategorialCrossEntropy loss;
    Vector z(3), y(3);
    z << 0.5, 0.3, 0.2;
    y << 1.0, 0.0, 0.0;
    // -1*log(0.5) = log(2)
    REQUIRE_THAT(loss.distance(z, y), WithinAbs(std::log(2.0), 1e-8));
}

TEST_CASE("CategorialCrossEntropy: нулевые метки не вносят вклад", "[loss][cce]")
{
    CategorialCrossEntropy loss;
    Vector z1(2), z2(2), y(2);
    z1 << 0.5, 0.3;
    z2 << 0.5, 0.9;   // отличается только нулевой компонент по y
    y  << 1.0, 0.0;
    REQUIRE_THAT(loss.distance(z1, y), WithinAbs(loss.distance(z2, y), 1e-10));
}

TEST_CASE("CategorialCrossEntropy: устойчивость при z близком к нулю", "[loss][cce]")
{
    CategorialCrossEntropy loss;
    Vector z(2), y(2);
    z << 0.0, 1.0;
    y << 0.0, 1.0;
    REQUIRE(std::isfinite(loss.distance(z, y)));
}

TEST_CASE("CategorialCrossEntropy: градиент совпадает с численным", "[loss][cce]")
{
    CategorialCrossEntropy loss;
    Vector z(3), y(3);
    z << 0.5, 0.3, 0.2;
    y << 1.0, 0.0, 0.0;
    check_gradient(loss, z, y, 1e-4);
}

TEST_CASE("CategorialCrossEntropy: градиент нулевой для нулевых меток", "[loss][cce]")
{
    CategorialCrossEntropy loss;
    Vector z(3), y(3);
    z << 0.5, 0.3, 0.2;
    y << 1.0, 0.0, 0.0;
    RowVector grad = loss.gradient(z, y);
    REQUIRE_THAT(grad[1], WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(grad[2], WithinAbs(0.0, 1e-12));
}
