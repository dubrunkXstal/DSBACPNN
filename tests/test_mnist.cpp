#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>

#include <bbx/blackbox/BlackBox.hpp>

// Датасет: https://github.com/phoebetronic/mnist

TEST_CASE("MNIST dataset", "[integration][mnist]")
{

    std::ifstream mnist_test("/Users/dobr.senuta/Downloads/mnist_test.csv");
    REQUIRE(mnist_test.is_open());

    std::string line;
    std::stringstream ss;

    bbx::BlackBox bb{
        {784, 500, bbx::Sigmoid{}},
        {500, 100, bbx::Sigmoid{}},
        {100, 100, bbx::Relu{}},
        {100, 200, bbx::Sigmoid{}},
        {200, 100, bbx::Sigmoid{}},
        {100, 10, bbx::Sigmoid{}}
    };

    REQUIRE(bb.getBlocksCount() == 6);

    int epochs_cnt = 1;
    int sample_size = 60;  // Сам MNIST по размеру - 60000.

    for (int e = 1; e <= epochs_cnt; ++e) {
        std::ifstream mnist_sample("/Users/dobr.senuta/Downloads/mnist_train.csv");
        REQUIRE(mnist_sample.is_open());

        std::string line;
        std::stringstream ss;
        std::string integer;
        int i = 0;

        while (mnist_sample.good() && std::getline(mnist_sample, line) && i < sample_size) {
            std::cout << "epoch " << e << "/" << epochs_cnt << " : - " << ++i << " -\n";

            ss = std::stringstream(line);
            std::getline(ss, integer, ',');

            Eigen::VectorXd y = Eigen::VectorXd::Zero(10);
            y[std::stoi(integer)] = 1;

            Eigen::VectorXd x(784);
            for (double& i : x) {
                std::getline(ss, integer, ',');
                i = std::stoi(integer) / 255.0;
            }

            bb.tuning(x, y);
        }

        mnist_sample.close();
    }

    std::string integer;
    int success_cnt = 0;

    while (mnist_test.good() && std::getline(mnist_test, line)) {
        ss = std::stringstream(line);
        std::getline(ss, integer, ',');

        Eigen::VectorXd y = Eigen::VectorXd::Zero(10);
        int y_int = std::stoi(integer);
        y[y_int] = 1;
        CAPTURE(integer);

        Eigen::VectorXd x(784);
        for (double& i : x) {
            std::getline(ss, integer, ',');
            i = std::stoi(integer) / 255.0;
        }

        Eigen::VectorXd res = bb.evaluate(x);

        int ind_max = 0;
        for (int i = 1; i < 10; ++i) {
            if (res[i] > res[ind_max]) {
                ind_max = i;
            }
        }

        CAPTURE(ind_max);

        if (ind_max == y_int) {
            ++success_cnt;
        }
    }

    CAPTURE(success_cnt);
    REQUIRE(success_cnt > 8000);
}
