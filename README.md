# bbx — Black Box

Header-only C++20 библиотека для построения и обучения полносвязных нейронных сетей.

Библиотека позволяет описать архитектуру сети буквально в несколько строк: задать количество слоёв, размерности, функции активации, оптимизатор и инициализации весов — и сразу начать обучение на данных из CSV.

## Зависимости

- Компилятор с поддержкой **C++20**
- **CMake 3.20+** (для сборки тестов или подключения через CMake)
- **Eigen 5.0.1** — единственная внешняя зависимость; при сборке через CMake скачивается автоматически через FetchContent

Для запуска тестов дополнительно используется **Catch2 3.8.1** (тоже подтягивается сам).

## Подключение

### Напрямую

Библиотека header-only, поэтому достаточно скопировать папку `include/bbx/` к себе в проект, убедиться, что Eigen доступен в include path, и подключить зонтичный заголовок:

```cpp
#include <bbx/bbx.hpp>
```

Компилировать со стандартом C++20:

```bash
g++ -std=c++20 -I path/to/include -I path/to/eigen main.cpp -o main
```

### Через CMake

Если ваш проект использует CMake, добавьте bbx через `FetchContent` или `add_subdirectory`:

- Вариант 1: FetchContent
```cmake
include(FetchContent)
FetchContent_Declare(
    bbx
    GIT_REPOSITORY <url-репозитория>
    GIT_TAG        main
)
FetchContent_MakeAvailable(bbx)

target_link_libraries(my_target PRIVATE bbx::bbx)
```

- Вариант 2: как подпапка `bbx`
```cmake
add_subdirectory(bbx)
target_link_libraries(my_target PRIVATE bbx::bbx)
```

CMake сам подтянет Eigen и выставит нужный стандарт компиляции.

## Возможности

### Функции активации

Sigmoid, ReLU, SELU, GELU, Tanh, Swish, ELU, LReLU, Softplus, Softsign, HardSigmoid, Exponential, Linear — всего 13 штук. У некоторых (Linear, LReLU, ELU, Swish, SELU) можно задать параметры через конструктор.

### Функции потерь

| Класс | Что считает |
|---|---|
| `L2NormSquared` | Сумма квадратов разностей (MSE) |
| `AbsoluteError` | Сумма модулей разностей (MAE) |
| `HuberLoss` | Huber loss (есть параметр delta) |
| `LogCosh` | Логарифм гиперболического косинуса |
| `BinaryCrossEntropy` | Бинарная кросс-энтропия |
| `CategorialCrossEntropy` | Категориальная кросс-энтропия |

### Оптимизаторы

- `VanillaDescent` — классический градиентный спуск
- `MomentumDescent` — с импульсом (настраиваемый beta)
- `Adam` — адаптивный момент (beta1, beta2, eps)

Каждый оптимизатор принимает расписание скорости обучения (`LRSchedule`): `ConstantLR` — постоянная, `TimeDecayLR` — затухание по мере обучения.

### Инициализация весов

`GlorotUniform`, `GlorotNormal`, `HeNormal`, `HeUniform`, `LeCunNormal`, `UniformRandom`, `Zeros` — выбираются в зависимости от функции активации. По умолчанию используется `GlorotUniform`.

### Работа с данными

Встроенная загрузка CSV через методы `loadTrainCSV` и `loadTestCSV` — с поддержкой мини-батчей, разделителей и заголовков.

### Гибкая конфигурация

Функция активации, оптимизатор и инициализатор весов задаются для каждого слоя отдельно. Можно менять их и после создания сети через `setActivation`, `setOptimizer`, `initializeWeights`.

Полиморфизм компонентов реализован через стирающие типы — без виртуальных базовых классов и накладных расходов на наследование.

## Быстрый старт

```cpp
#include <bbx/bbx.hpp>

int main() {
    // Сеть: 784 -> 500 -> 100 -> 10
    bbx::BlackBox network{
        {784, 500, bbx::SELU{}, bbx::Adam{}, bbx::LeCunNormal{}},
        {500, 100, bbx::SELU{}, bbx::Adam{}, bbx::LeCunNormal{}},
        {100,  10, bbx::Sigmoid{}, bbx::Adam{}}
    };

    network.setLoss(bbx::BinaryCrossEntropy{});

    // Обучение — первые 10 столбцов CSV считаются целевыми
    network.loadTrainCSV("train.csv", {0,1,2,3,4,5,6,7,8,9});

    // Оценка точности. Интерпретация выхода сети – задача пользователя
    double accuracy = network.loadTestCSV(
        "test.csv",
        {0,1,2,3,4,5,6,7,8,9},
        [](const bbx::Vector& predicted, const bbx::Vector& actual) {
            bbx::Index p, a;
            predicted.maxCoeff(&p);
            actual.maxCoeff(&a);
            return p == a;
        }
    );
}
```

## Сборка проекта и тесты

```bash
cmake -S . -B build
cmake --build build
```

```bash
ctest --test-dir build  # все тесты
ctest --test-dir build -L unit  # только unit-тесты
ctest --test-dir build -L integration --output-on-failure  # только интеграционные
```
