#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bbx/bbx.hpp>
#include <cmath>

using namespace bbx;
using Catch::Matchers::WithinAbs;

// Вычисляет среднеквадратичное отклонение матрицы
static double sample_std(const Matrix& m)
{
    double mean = m.mean();
    return std::sqrt((m.array() - mean).square().mean());
}

// Zeros

TEST_CASE("Zeros: все элементы равны нулю", "[initializer][zeros]")
{
    Zeros init;
    Matrix m = init.generate(5, 10);
    REQUIRE(m.isZero(1e-12));
}

TEST_CASE("Zeros: правильные размерности", "[initializer][zeros]")
{
    Zeros init;
    Matrix m = init.generate(7, 3);
    REQUIRE(m.rows() == 7);
    REQUIRE(m.cols() == 3);
}

// UniformRandom

TEST_CASE("UniformRandom: правильные размерности", "[initializer][uniform]")
{
    UniformRandom init;
    Matrix m = init.generate(7, 3);
    REQUIRE(m.rows() == 7);
    REQUIRE(m.cols() == 3);
}

TEST_CASE("UniformRandom: значения лежат в [-1, 1]", "[initializer][uniform]")
{
    UniformRandom init;
    Matrix m = init.generate(100, 100);
    REQUIRE(m.minCoeff() >= -1.0);
    REQUIRE(m.maxCoeff() <=  1.0);
}

// GlorotUniform

TEST_CASE("GlorotUniform: правильные размерности", "[initializer][glorot]")
{
    GlorotUniform init;
    Matrix m = init.generate(4, 8);
    REQUIRE(m.rows() == 4);
    REQUIRE(m.cols() == 8);
}

TEST_CASE("GlorotUniform: значения в пределах [-limit, limit]", "[initializer][glorot]")
{
    GlorotUniform init;
    int rows = 4, cols = 8;
    Matrix m = init.generate(rows, cols);
    double limit = std::sqrt(6.0 / (rows + cols));
    REQUIRE(m.minCoeff() >= -limit - 1e-12);
    REQUIRE(m.maxCoeff() <=  limit + 1e-12);
}

TEST_CASE("GlorotUniform: среднее близко к нулю для большой матрицы", "[initializer][glorot]")
{
    GlorotUniform init;
    Matrix m = init.generate(500, 200);
    REQUIRE_THAT(m.mean(), WithinAbs(0.0, 0.02));
}

// GlorotNormal

TEST_CASE("GlorotNormal: правильные размерности", "[initializer][glorot]")
{
    GlorotNormal init;
    Matrix m = init.generate(6, 9);
    REQUIRE(m.rows() == 6);
    REQUIRE(m.cols() == 9);
}

TEST_CASE("GlorotNormal: среднее близко к нулю", "[initializer][glorot]")
{
    GlorotNormal init;
    Matrix m = init.generate(1000, 100);
    REQUIRE_THAT(m.mean(), WithinAbs(0.0, 0.02));
}

TEST_CASE("GlorotNormal: правильное стандартное отклонение", "[initializer][glorot]")
{
    GlorotNormal init;
    int rows = 1000, cols = 100;
    Matrix m = init.generate(rows, cols);
    double expected = std::sqrt(2.0 / (cols + rows));
    REQUIRE_THAT(sample_std(m), WithinAbs(expected, expected * 0.1));
}

// HeNormal

TEST_CASE("HeNormal: правильные размерности", "[initializer][he]")
{
    HeNormal init;
    Matrix m = init.generate(8, 5);
    REQUIRE(m.rows() == 8);
    REQUIRE(m.cols() == 5);
}

TEST_CASE("HeNormal: среднее близко к нулю", "[initializer][he]")
{
    HeNormal init;
    Matrix m = init.generate(1000, 100);
    REQUIRE_THAT(m.mean(), WithinAbs(0.0, 0.02));
}

TEST_CASE("HeNormal: правильное стандартное отклонение", "[initializer][he]")
{
    HeNormal init;
    int cols = 100;
    Matrix m = init.generate(1000, cols);
    double expected = std::sqrt(2.0 / cols);
    REQUIRE_THAT(sample_std(m), WithinAbs(expected, expected * 0.1));
}

// HeUniform

TEST_CASE("HeUniform: значения в пределах [-limit, limit]", "[initializer][he]")
{
    HeUniform init;
    int cols = 10;
    Matrix m = init.generate(100, cols);
    double limit = std::sqrt(6.0 / cols);
    REQUIRE(m.minCoeff() >= -limit - 1e-12);
    REQUIRE(m.maxCoeff() <=  limit + 1e-12);
}

// LeCunNormal

TEST_CASE("LeCunNormal: правильные размерности", "[initializer][lecun]")
{
    LeCunNormal init;
    Matrix m = init.generate(3, 12);
    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 12);
}

TEST_CASE("LeCunNormal: среднее близко к нулю", "[initializer][lecun]")
{
    LeCunNormal init;
    Matrix m = init.generate(1000, 100);
    REQUIRE_THAT(m.mean(), WithinAbs(0.0, 0.02));
}

TEST_CASE("LeCunNormal: правильное стандартное отклонение", "[initializer][lecun]")
{
    LeCunNormal init;
    int cols = 100;
    Matrix m = init.generate(1000, cols);
    double expected = std::sqrt(1.0 / cols);
    REQUIRE_THAT(sample_std(m), WithinAbs(expected, expected * 0.1));
}

TEST_CASE("LeCunNormal: отклонение меньше, чем у HeNormal", "[initializer][lecun]")
{
    LeCunNormal lecun;
    HeNormal    he;
    int cols = 50;
    // LeCun: std=sqrt(1/n), He: std=sqrt(2/n), значит LeCun < He
    REQUIRE(sample_std(lecun.generate(500, cols)) <
            sample_std(he.generate(500, cols)) * 1.5);
}
