#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bbx/bbx.hpp>
#include <cmath>
#include <functional>

using namespace bbx;
using Catch::Matchers::WithinAbs;

// Центральная разность для проверки производной
static double num_deriv(std::function<double(double)> f, double x)
{
    constexpr double h = 1e-5;
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

// Linear

TEST_CASE("Linear: вычисление значений", "[activation][linear]")
{
    Linear f{2.0, 3.0};
    REQUIRE_THAT(f.evaluate(0.0),  WithinAbs(3.0,  1e-12));
    REQUIRE_THAT(f.evaluate(1.0),  WithinAbs(5.0,  1e-12));
    REQUIRE_THAT(f.evaluate(-2.0), WithinAbs(-1.0, 1e-12));
}

TEST_CASE("Linear: производная постоянна и равна alpha", "[activation][linear]")
{
    Linear f{2.0, 3.0};
    for (double x : {-3.0, 0.0, 1.0, 5.0}) {
        REQUIRE_THAT(f.derivative(x), WithinAbs(2.0, 1e-12));
    }
}

// Sigmoid

TEST_CASE("Sigmoid: значения в известных точках", "[activation][sigmoid]")
{
    Sigmoid f;
    REQUIRE_THAT(f.evaluate(0.0),   WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(f.evaluate(100.0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(f.evaluate(-100.), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Sigmoid: выход лежит в интервале (0, 1)", "[activation][sigmoid]")
{
    Sigmoid f;
    for (double x : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
        REQUIRE(f.evaluate(x) > 0.0);
        REQUIRE(f.evaluate(x) < 1.0);
    }
}

TEST_CASE("Sigmoid: производная совпадает с численной", "[activation][sigmoid]")
{
    Sigmoid f;
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
        REQUIRE_THAT(f.derivative(x),
                     WithinAbs(num_deriv([&](double v){ return f.evaluate(v); }, x), 1e-6));
    }
}

// HardSigmoid

TEST_CASE("HardSigmoid: насыщение на краях", "[activation][hardsigmoid]")
{
    HardSigmoid f;
    REQUIRE_THAT(f.evaluate(-3.0), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(f.evaluate(3.0),  WithinAbs(1.0, 1e-12));
}

TEST_CASE("HardSigmoid: линейная область", "[activation][hardsigmoid]")
{
    HardSigmoid f;
    REQUIRE_THAT(f.evaluate(0.0), WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(f.evaluate(1.0), WithinAbs(0.7, 1e-12));
}

TEST_CASE("HardSigmoid: производная", "[activation][hardsigmoid]")
{
    HardSigmoid f;
    REQUIRE_THAT(f.derivative(0.0),  WithinAbs(0.2, 1e-12));
    REQUIRE_THAT(f.derivative(-3.0), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(f.derivative(3.0),  WithinAbs(0.0, 1e-12));
}

// ReLU

TEST_CASE("ReLU: вычисление значений", "[activation][relu]")
{
    ReLU f;
    REQUIRE_THAT(f.evaluate(2.0),  WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(f.evaluate(-2.0), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(f.evaluate(0.0),  WithinAbs(0.0, 1e-12));
}

TEST_CASE("ReLU: производная", "[activation][relu]")
{
    ReLU f;
    REQUIRE_THAT(f.derivative(1.0),  WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(f.derivative(-1.0), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(f.derivative(0.0),  WithinAbs(1.0, 1e-12));
}

// LReLU

TEST_CASE("LReLU: вычисление значений", "[activation][lrelu]")
{
    LReLU f{0.1};
    REQUIRE_THAT(f.evaluate(2.0),  WithinAbs(2.0,  1e-12));
    REQUIRE_THAT(f.evaluate(-2.0), WithinAbs(-0.2, 1e-12));
    REQUIRE_THAT(f.evaluate(0.0),  WithinAbs(0.0,  1e-12));
}

TEST_CASE("LReLU: производная", "[activation][lrelu]")
{
    LReLU f{0.1};
    REQUIRE_THAT(f.derivative(1.0),  WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(f.derivative(-1.0), WithinAbs(0.1, 1e-12));
}

// ELU

TEST_CASE("ELU: вычисление значений", "[activation][elu]")
{
    ELU f{1.0};
    REQUIRE_THAT(f.evaluate(2.0), WithinAbs(2.0,                        1e-12));
    REQUIRE_THAT(f.evaluate(0.0), WithinAbs(0.0,                        1e-12));
    REQUIRE_THAT(f.evaluate(-1.), WithinAbs(std::exp(-1.0) - 1.0,       1e-10));
}

TEST_CASE("ELU: непрерывность в нуле", "[activation][elu]")
{
    ELU f{1.0};
    REQUIRE_THAT(f.evaluate(1e-9), WithinAbs(f.evaluate(-1e-9), 1e-7));
}

TEST_CASE("ELU: производная совпадает с численной", "[activation][elu]")
{
    ELU f{1.0};
    for (double x : {-3.0, -1.0, -0.1, 1.0, 3.0}) {
        REQUIRE_THAT(f.derivative(x),
                     WithinAbs(num_deriv([&](double v){ return f.evaluate(v); }, x), 1e-6));
    }
}

// SELU

TEST_CASE("SELU: значение в нуле", "[activation][selu]")
{
    SELU f;
    REQUIRE_THAT(f.evaluate(0.0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("SELU: масштабирование для положительных входов", "[activation][selu]")
{
    SELU f;
    REQUIRE_THAT(f.evaluate(1.0), WithinAbs(1.05070098, 1e-7));
}

TEST_CASE("SELU: производная совпадает с численной", "[activation][selu]")
{
    SELU f;
    for (double x : {-3.0, -1.0, 1.0, 3.0}) {
        REQUIRE_THAT(f.derivative(x),
                     WithinAbs(num_deriv([&](double v){ return f.evaluate(v); }, x), 1e-6));
    }
}

// GELU

TEST_CASE("GELU: значение в нуле", "[activation][gelu]")
{
    GELU f;
    REQUIRE_THAT(f.evaluate(0.0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("GELU: производная в нуле равна 0.5", "[activation][gelu]")
{
    GELU f;
    REQUIRE_THAT(f.derivative(0.0), WithinAbs(0.5, 1e-6));
}

TEST_CASE("GELU: производная совпадает с численной", "[activation][gelu]")
{
    GELU f;
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        REQUIRE_THAT(f.derivative(x),
                     WithinAbs(num_deriv([&](double v){ return f.evaluate(v); }, x), 1e-4));
    }
}

// Exponential

TEST_CASE("Exponential: значение и производная равны exp(x)", "[activation][exp]")
{
    Exponential f;
    REQUIRE_THAT(f.evaluate(0.0),    WithinAbs(1.0,             1e-12));
    REQUIRE_THAT(f.evaluate(1.0),    WithinAbs(std::exp(1.0),   1e-10));
    REQUIRE_THAT(f.derivative(1.0),  WithinAbs(std::exp(1.0),   1e-10));
    REQUIRE_THAT(f.derivative(0.0),  WithinAbs(1.0,             1e-12));
}

// Swish

TEST_CASE("Swish: значение в нуле", "[activation][swish]")
{
    Swish f;
    REQUIRE_THAT(f.evaluate(0.0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Swish: производная в нуле равна 0.5", "[activation][swish]")
{
    Swish f;
    REQUIRE_THAT(f.derivative(0.0), WithinAbs(0.5, 1e-10));
}

TEST_CASE("Swish: производная совпадает с численной", "[activation][swish]")
{
    Swish f;
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
        REQUIRE_THAT(f.derivative(x),
                     WithinAbs(num_deriv([&](double v){ return f.evaluate(v); }, x), 1e-5));
    }
}

TEST_CASE("Swish: производная конечна и близка к 1 при большом x", "[activation][swish]")
{
    Swish f;
    double d = f.derivative(1000.0);
    REQUIRE(std::isfinite(d));
    REQUIRE_THAT(d, WithinAbs(1.0, 1e-6));
}

// Softplus

TEST_CASE("Softplus: значение в нуле равно ln(2)", "[activation][softplus]")
{
    Softplus f;
    REQUIRE_THAT(f.evaluate(0.0), WithinAbs(std::log(2.0), 1e-10));
}

TEST_CASE("Softplus: устойчивость при большом x", "[activation][softplus]")
{
    Softplus f;
    REQUIRE_THAT(f.evaluate(25.0), WithinAbs(25.0, 1e-6));
}

TEST_CASE("Softplus: производная совпадает с численной", "[activation][softplus]")
{
    Softplus f;
    for (double x : {-3.0, 0.0, 1.0, 3.0}) {
        REQUIRE_THAT(f.derivative(x),
                     WithinAbs(num_deriv([&](double v){ return f.evaluate(v); }, x), 1e-5));
    }
}

// Softsign

TEST_CASE("Softsign: значения в известных точках", "[activation][softsign]")
{
    Softsign f;
    REQUIRE_THAT(f.evaluate(0.0),  WithinAbs(0.0,  1e-12));
    REQUIRE_THAT(f.evaluate(1.0),  WithinAbs(0.5,  1e-12));
    REQUIRE_THAT(f.evaluate(-1.0), WithinAbs(-0.5, 1e-12));
}

TEST_CASE("Softsign: выход ограничен интервалом (-1, 1)", "[activation][softsign]")
{
    Softsign f;
    for (double x : {-100.0, -1.0, 0.0, 1.0, 100.0}) {
        REQUIRE(f.evaluate(x) > -1.0);
        REQUIRE(f.evaluate(x) <  1.0);
    }
}

TEST_CASE("Softsign: производная совпадает с численной", "[activation][softsign]")
{
    Softsign f;
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
        REQUIRE_THAT(f.derivative(x),
                     WithinAbs(num_deriv([&](double v){ return f.evaluate(v); }, x), 1e-5));
    }
}

// Tanh

TEST_CASE("Tanh: значения в известных точках", "[activation][tanh]")
{
    Tanh f;
    REQUIRE_THAT(f.evaluate(0.0), WithinAbs(0.0,              1e-12));
    REQUIRE_THAT(f.evaluate(1.0), WithinAbs(std::tanh(1.0),   1e-12));
}

TEST_CASE("Tanh: выход ограничен интервалом (-1, 1)", "[activation][tanh]")
{
    Tanh f;
    REQUIRE(f.evaluate(5.0)  <  1.0);
    REQUIRE(f.evaluate(-5.0) > -1.0);
}

TEST_CASE("Tanh: производная совпадает с численной", "[activation][tanh]")
{
    Tanh f;
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
        REQUIRE_THAT(f.derivative(x),
                     WithinAbs(num_deriv([&](double v){ return f.evaluate(v); }, x), 1e-5));
    }
}
