#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "BlackBox.h"
#include "eigen/Eigen/Core"
#include "eigen/Eigen/Dense"

int main() {
    std::ifstream settings("settings.txt");
    assert(settings.is_open());

    std::ifstream mnist_test("MNIST/mnist_test.csv");
    assert(mnist_test.is_open());
    {
        std::ifstream mnist_sample("MNIST/mnist_train.csv");
        assert(mnist_sample.is_open());
        mnist_sample.close();
    }

    std::string line;
    std::getline(settings, line);
    std::stringstream ss(line);

    int blocks_cnt;
    ss >> blocks_cnt;
    BlackBox bb(blocks_cnt, settings);

    int epochs_cnt = 3;
    int sample_size = 60000;  // Сам MNIST по размеру - 60000.

    for (int e = 1; e <= epochs_cnt; ++e) {
        std::ifstream mnist_sample("MNIST/mnist_train.csv");
        assert(mnist_sample.is_open());

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

    std::cout << "--- after tuning ---\n";

    std::string integer;
    std::string go;
    int success_cnt = 0;

    while (mnist_test.good() && std::getline(mnist_test, line)) {
        if (go != "go") {
            std::cin >> go;
        }

        ss = std::stringstream(line);
        std::getline(ss, integer, ',');

        Eigen::VectorXd y = Eigen::VectorXd::Zero(10);
        int y_int = std::stoi(integer);
        y[y_int] = 1;
        std::cout << "Referen: " << integer << '\n';

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

        std::cout << "Predict: " << ind_max << "\n----------\n";

        if (ind_max == y_int) {
            ++success_cnt;
        }
    }

    std::cout << "Success rate: " << success_cnt / 100.0 << "%\n";

    return 0;
}
