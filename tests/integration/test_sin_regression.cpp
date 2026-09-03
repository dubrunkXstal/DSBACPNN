#include <catch2/catch_test_macros.hpp>

#include <bbx/blackbox/BlackBox.hpp>
#include <cmath>

/*
Аппроксимация функции sin(x) на отрезке [-pi, pi].
Входы нормализованы в [-1, 1], выходы в [-1, 1].
Проверяет работу регрессии (L2NormSquared) и активации Linear на выходном слое.
*/

TEST_CASE("Sin regression", "[integration]")
{
    bbx::BlackBox bb{
        {1, 32, bbx::Tanh{}, bbx::Adam{}},
        {32, 32, bbx::Tanh{}, bbx::Adam{}},
        {32, 1, bbx::Linear{}, bbx::Adam{}}
    };

    bb.setLoss(bbx::L2NormSquared{});

    constexpr double pi = 3.14159265358979323846;
    constexpr int N = 100;

    std::vector<std::pair<bbx::Vector, bbx::Vector>> train_data;
    for (int i = 0; i < N; ++i) {
        double x_val = -pi + 2.0 * pi * i / (N - 1);
        bbx::Vector x(1); x << x_val / pi;
        bbx::Vector y(1); y << std::sin(x_val);
        train_data.push_back({x, y});
    }

    for (int epoch = 0; epoch < 5000; ++epoch) {
        for (auto& [x, y] : train_data) {
            bb.tuning(x, y);
        }
    }

    double mse = 0.0;
    constexpr int T = 50;
    for (int i = 0; i < T; ++i) {
        double x_val = -pi + 2.0 * pi * i / (T - 1);
        bbx::Vector x(1); x << x_val / pi;
        double predicted = bb.evaluate(x)(0);
        double expected = std::sin(x_val);
        mse += (predicted - expected) * (predicted - expected);
    }
    mse /= T;

    CAPTURE(mse);
    CHECK(mse < 0.05);
}
