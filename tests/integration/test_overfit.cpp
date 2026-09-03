#include <catch2/catch_test_macros.hpp>

#include <bbx/blackbox/BlackBox.hpp>

/*
Тест на переобучение: сеть должна запомнить маленький датасет (10 примеров).
Если loss не стремится к нулю — значит в backprop есть ошибка.
Косвенно верифицирует корректность вычисления градиентов.
*/

TEST_CASE("Overfit", "[integration]")
{
    bbx::BlackBox bb{
        {4, 32, bbx::SELU{}, bbx::Adam{}, bbx::LeCunNormal{}},
        {32, 32, bbx::SELU{}, bbx::Adam{}, bbx::LeCunNormal{}},
        {32, 3, bbx::Sigmoid{}, bbx::Adam{}}
    };

    bb.setLoss(bbx::BinaryCrossEntropy{});

    // 10 фиксированных примеров: 4 входа -> 3 выхода (one-hot)
    std::vector<std::pair<bbx::Vector, bbx::Vector>> data;

    auto make = [](std::initializer_list<double> xv, std::initializer_list<double> yv) {
        bbx::Vector x(xv.size()); int i = 0; for (double v : xv) x(i++) = v;
        bbx::Vector y(yv.size()); i = 0;     for (double v : yv) y(i++) = v;
        return std::make_pair(x, y);
    };

    data.push_back(make({0.1, 0.2, 0.3, 0.4}, {1, 0, 0}));
    data.push_back(make({0.5, 0.6, 0.7, 0.8}, {0, 1, 0}));
    data.push_back(make({0.9, 0.1, 0.2, 0.3}, {0, 0, 1}));
    data.push_back(make({0.4, 0.5, 0.6, 0.7}, {1, 0, 0}));
    data.push_back(make({0.8, 0.9, 0.1, 0.2}, {0, 1, 0}));
    data.push_back(make({0.3, 0.4, 0.5, 0.6}, {0, 0, 1}));
    data.push_back(make({0.7, 0.8, 0.9, 0.1}, {1, 0, 0}));
    data.push_back(make({0.2, 0.3, 0.4, 0.5}, {0, 1, 0}));
    data.push_back(make({0.6, 0.7, 0.8, 0.9}, {0, 0, 1}));
    data.push_back(make({0.0, 0.1, 0.9, 0.5}, {1, 0, 0}));

    for (int epoch = 0; epoch < 3000; ++epoch) {
        for (auto& [x, y] : data) {
            bb.tuning(x, y);
        }
    }

    int correct = 0;
    for (auto& [x, y] : data) {
        bbx::Vector pred = bb.evaluate(x);
        bbx::Index pred_argmax, y_argmax;
        pred.maxCoeff(&pred_argmax);
        y.maxCoeff(&y_argmax);
        if (pred_argmax == y_argmax) ++correct;
    }

    double accuracy = static_cast<double>(correct) / data.size();
    CAPTURE(accuracy);
    CHECK(accuracy == 1.0);
}
