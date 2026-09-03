#include <catch2/catch_test_macros.hpp>

#include <bbx/blackbox/BlackBox.hpp>

/*
Датасет: https://archive.ics.uci.edu/dataset/53/iris
Предварительно обработан:
- столбец класса заменён на 3 столбца по One Hot Encoding
- признаки отнормированы min-max в [0, 1]

Формат CSV после обработки:
    столбцы 0-2: one-hot класс
    столбцы 3-6: нормализованные признаки (sepal_len, sepal_wid, petal_len, petal_wid)

Скрипт для обработки:
```python
import csv

INPUT  = "iris.csv"
OUTPUT = "iris_processed.csv"

cls_map = {"Iris-setosa": [1,0,0], "Iris-versicolor": [0,1,0], "Iris-virginica": [0,0,1]}

rows = []
with open(INPUT, newline='') as f:
    for row in csv.reader(f):
        features = [float(v) for v in row[:4]]
        rows.append((features, row[4]))

mins = [min(r[0][i] for r in rows) for i in range(4)]
maxs = [max(r[0][i] for r in rows) for i in range(4)]

with open(OUTPUT, 'w', newline='') as f:
    writer = csv.writer(f)
    for features, label in rows:
        normed = [(features[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(4)]
        writer.writerow(cls_map[label] + normed)
```
*/

TEST_CASE("Iris", "[integration]")
{
    bbx::BlackBox bb{
        {4, 16, bbx::Tanh{}, bbx::Adam{}},
        {16, 16, bbx::Tanh{}, bbx::Adam{}},
        {16, 3, bbx::Sigmoid{}, bbx::Adam{}}
    };

    bb.setLoss(bbx::BinaryCrossEntropy{});

    REQUIRE(bb.getBlocksCount() == 3);

    bb.loadTrainCSV(
        "",
        {0, 1, 2}
    );

    double score = bb.loadTestCSV(
        "",
        {0, 1, 2},
        [](const bbx::Vector& x, const bbx::Vector& y) {
            bbx::Index x_argmax, y_argmax;
            x.maxCoeff(&x_argmax);
            y.maxCoeff(&y_argmax);
            return x_argmax == y_argmax;
    });

    CAPTURE(score);
    FAIL("[SUCCESS] Forcing failure to view captures");
}
