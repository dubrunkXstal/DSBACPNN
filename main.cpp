#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "EigenRand/EigenRand/EigenRand"
#include <memory>
#include <string>
#include <cstdio>
#include "BlackBox.h"


int main() {
    std::ifstream settings("settings.txt");
    if (!settings.is_open()) {
        std::cerr << "Error: Unable to open settings.txt!" << '\n';
        return 1;
    }

    std::ifstream mnist_test("MNIST/mnist_test.csv");
    if (!mnist_test.is_open()) {
        std::cerr << "Error: Unable to open mnist_test.txt!" << '\n';
        return 1;
    }
    {
        std::ifstream mnist_sample("MNIST/mnist_train.csv");
        if (!mnist_sample.is_open()) {
            std::cerr << "Error: Unable to open mnist_train.txt!" << '\n';
            return 1;
        }
        mnist_sample.close();
    }

    std::string line;
    std::getline(settings, line);
    std::stringstream ss(line);

    size_t blocks_cnt;
    ss >> blocks_cnt;
    BlackBox bb(blocks_cnt, settings);

    size_t epochs_cnt = 8;
    size_t sample_size = 60000;  // Сам MNIST по размеру - 60000.

    for (int e = 1; e <= epochs_cnt; ++e) {
        std::ifstream mnist_sample("MNIST/mnist_train.csv");
        if (!mnist_sample.is_open()) {
            std::cerr << "Error: Unable to open mnist_train.txt!" << '\n';
            return 1;
        }

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
    size_t success_cnt = 0;

    while (mnist_test.good() && std::getline(mnist_test, line)) {
        if (go != "go") { std::cin >> go; }

        ss = std::stringstream(line);
        std::getline(ss, integer, ',');

        Eigen::VectorXd y = Eigen::VectorXd::Zero(10);
        y[std::stoi(integer)] = 1;
        std::cout << "Referen: " << integer << '\n';
        int y_int = std::stoi(integer);

        Eigen::VectorXd x(784);
        for (double& i : x) {
            std::getline(ss, integer, ',');
            i = std::stoi(integer) / 255.0;
        }

         Eigen::VectorXd res = bb.evaluate(x);

         int ind_max = 0;
         for (int i = 1; i < 10; ++i) {
             if (res[i] > res[ind_max]) { ind_max = i; }
         }

         std::cout << "Predict: " << ind_max << "\n----------\n";

         if (ind_max == y_int) { ++success_cnt; }
    }

    std::cout << "Success rate: " << success_cnt / 100.0 << "%\n";

    return 0;
}
