#include <catch2/catch_test_macros.hpp>

#include <bbx/blackbox/BlackBox.hpp>

TEST_CASE("XOR", "[integration]")
{
    bbx::BlackBox bb{
        {2, 8, bbx::Tanh{}, bbx::Adam{}},
        {8, 1, bbx::Sigmoid{}, bbx::Adam{}}
    };

    bb.setLoss(bbx::BinaryCrossEntropy{});

    bbx::Vector x00(2); x00 << 0, 0;
    bbx::Vector x01(2); x01 << 0, 1;
    bbx::Vector x10(2); x10 << 1, 0;
    bbx::Vector x11(2); x11 << 1, 1;

    bbx::Vector y0(1); y0 << 0;
    bbx::Vector y1(1); y1 << 1;

    std::vector<std::pair<bbx::Vector, bbx::Vector>> data = {
        {x00, y0}, {x01, y1}, {x10, y1}, {x11, y0}
    };

    for (int epoch = 0; epoch < 5000; ++epoch) {
        for (auto& [x, y] : data) {
            bb.tuning(x, y);
        }
    }

    double out00 = bb.evaluate(x00)(0);
    double out01 = bb.evaluate(x01)(0);
    double out10 = bb.evaluate(x10)(0);
    double out11 = bb.evaluate(x11)(0);

    CAPTURE(out00, out01, out10, out11);

    CHECK(out00 < 0.1);
    CHECK(out01 > 0.9);
    CHECK(out10 > 0.9);
    CHECK(out11 < 0.1);
}
