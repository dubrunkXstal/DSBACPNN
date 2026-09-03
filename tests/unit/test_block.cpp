#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bbx/bbx.hpp>

using namespace bbx;
using Catch::Matchers::WithinAbs;

// Размерности

TEST_CASE("Block: evaluate возвращает вектор правильного размера", "[block]")
{
    Block block(4, 7, Sigmoid{});
    Vector x = Vector::Ones(4);
    REQUIRE(block.evaluate(x).size() == 7);
}

TEST_CASE("Block: пакетный evaluate возвращает матрицу правильной формы", "[block]")
{
    Block block(3, 5, ReLU{});
    Matrix batch = Matrix::Random(3, 8);
    Matrix out = block.evaluate(batch);
    REQUIRE(out.rows() == 5);
    REQUIRE(out.cols() == 8);
}

// Инициализация смещений

TEST_CASE("Block: смещения инициализируются нулями", "[block]")
{
    // evaluate(0) = sigma(A*0 + b) = sigma(b)
    // Linear(alpha=1, beta=0): sigma(b) = b
    // Если b=0, то evaluate(0) = 0
    Block block(5, 3, Linear{1.0, 0.0});
    Vector x = Vector::Zero(5);
    REQUIRE(block.evaluate(x).isZero(1e-12));
}

// Пользовательский инициализатор

TEST_CASE("Block: нулевой инициализатор делает все выходы одинаковыми", "[block]")
{
    // A=0, b=0, значит evaluate(x) = sigma(0) для любого x
    Block block(5, 3, Sigmoid{}, VanillaDescent{}, Zeros{});
    Vector x1 = Vector::Ones(5);
    Vector x2 = Vector::Random(5);
    // Sigmoid(0) = 0.5 для каждого выхода
    REQUIRE(block.evaluate(x1).isApprox(block.evaluate(x2)));
    for (int i = 0; i < 3; ++i) {
        REQUIRE_THAT(block.evaluate(x1)[i], WithinAbs(0.5, 1e-10));
    }
}

// Свойства выходов в зависимости от активации

TEST_CASE("Block: Sigmoid дает выход строго в (0, 1)", "[block]")
{
    Block block(4, 6, Sigmoid{});
    Vector x = Vector::Random(4);
    Vector out = block.evaluate(x);
    REQUIRE((out.array() > 0.0).all());
    REQUIRE((out.array() < 1.0).all());
}

TEST_CASE("Block: ReLU дает неотрицательный выход", "[block]")
{
    Block block(4, 6, ReLU{});
    Vector x = Vector::Random(4);
    REQUIRE((block.evaluate(x).array() >= 0.0).all());
}

TEST_CASE("Block: Tanh дает выход строго в (-1, 1)", "[block]")
{
    Block block(4, 6, Tanh{});
    Vector x = Vector::Random(4);
    Vector out = block.evaluate(x);
    REQUIRE((out.array() > -1.0).all());
    REQUIRE((out.array() <  1.0).all());
}

// Пакетное вычисление

TEST_CASE("Block: пакетный evaluate совпадает с поэлементным", "[block]")
{
    Block block(3, 4, Sigmoid{});
    Vector x1 = Vector::Random(3);
    Vector x2 = Vector::Random(3);
    Vector x3 = Vector::Random(3);

    Matrix batch(3, 3);
    batch.col(0) = x1;
    batch.col(1) = x2;
    batch.col(2) = x3;

    Matrix batch_out = block.evaluate(batch);
    REQUIRE(batch_out.col(0).isApprox(block.evaluate(x1)));
    REQUIRE(batch_out.col(1).isApprox(block.evaluate(x2)));
    REQUIRE(batch_out.col(2).isApprox(block.evaluate(x3)));
}

// Градиентный спуск

TEST_CASE("Block: gradientDescent изменяет выход", "[block]")
{
    Block block(3, 2, Sigmoid{});
    Vector x = Vector::Random(3);

    Vector before = block.evaluate(x);

    // Сигнал обратного распространения: u = (z - 0)^T
    RowVector u = block.evaluate(x).transpose();
    block.gradientDescent(x, u);

    Vector after = block.evaluate(x);
    REQUIRE_FALSE(before.isApprox(after));
}

TEST_CASE("Block: пакетный gradientDescent изменяет выход", "[block]")
{
    Block block(3, 2, Sigmoid{});

    Matrix x_batch = Matrix::Random(3, 4);
    Matrix before = block.evaluate(x_batch);

    // u_batch: (n_out x n_samples)
    Matrix u_batch = block.evaluate(x_batch).transpose();
    block.gradientDescent(x_batch, u_batch);

    Matrix after = block.evaluate(x_batch);
    REQUIRE_FALSE(before.isApprox(after));
}

TEST_CASE("Block: выход конечен после нескольких шагов gradientDescent", "[block]")
{
    Block block(5, 3, Sigmoid{}, Adam{}, GlorotUniform{});
    Vector x = Vector::Random(5);

    for (int step = 0; step < 10; ++step) {
        RowVector u = block.evaluate(x).transpose();
        block.gradientDescent(x, u);
    }

    Vector out = block.evaluate(x);
    REQUIRE(out.array().isFinite().all());
}
