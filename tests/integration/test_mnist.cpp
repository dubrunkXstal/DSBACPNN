#include <catch2/catch_test_macros.hpp>

#include <bbx/blackbox/BlackBox.hpp>

/*
Датасет: https://github.com/phoebetronic/mnist
Предварительно обработан:
- столбец цифры заменён на 10 столбцов по One Hot Encoding
- пиксели отнормированы по 255.0

Скрипт для обработки:
```python
import csv

INPUT  = "mnist_train.csv"
OUTPUT = "mnist_train_processed.csv"

with open(INPUT, newline='') as fin, open(OUTPUT, 'w', newline='') as fout:
    writer = csv.writer(fout)
    for row in csv.reader(fin):
        label  = int(row[0])
        pixels = [int(v) / 255.0 for v in row[1:]]

        one_hot = [0.0] * 10
        one_hot[label] = 1.0

        writer.writerow(one_hot + pixels)
```
*/

TEST_CASE("MNIST", "[integration]")
{
    bbx::BlackBox bb{
        {784, 500, bbx::SELU{}, bbx::Adam{}, bbx::LeCunNormal{}},
        {500, 100, bbx::SELU{}, bbx::Adam{}, bbx::LeCunNormal{}},
        {100, 100, bbx::SELU{}, bbx::Adam{}, bbx::LeCunNormal{}},
        {100, 200, bbx::SELU{}, bbx::Adam{}, bbx::LeCunNormal{}},
        {200, 100, bbx::SELU{}, bbx::Adam{}, bbx::LeCunNormal{}},
        {100, 10, bbx::Sigmoid{}, bbx::Adam{}}
    };

    bb.setLoss(bbx::BinaryCrossEntropy{});

    REQUIRE(bb.getBlocksCount() == 6);

    bb.loadTrainCSV(
        "",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    );

    double score = 0.0;
    score = bb.loadTestCSV(
        "",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        [](const bbx::Vector& x, const bbx::Vector& y) {
            bbx::Index x_argmax, y_argmax;
            x.maxCoeff(&x_argmax);
            y.maxCoeff(&y_argmax);
            return x_argmax == y_argmax;
    });

    CAPTURE(score);
    FAIL("[SUCCESS] Forcing failure to view captures");
}
