#include <random>
#include <iostream>


int main() {
    int in_dim;
    int out_dim;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.5, 1);

    while (true) {
        std::cout << "Input dimensions of the block (in_dim, out_dim): ";
        std::cin >> in_dim >> out_dim;


        std::cout << "\nTheta for dimensions (" << in_dim << ", " << out_dim << "):\n";

        for (int i = 0; i < out_dim * (in_dim + 1); ++i) {
            std::cout << distribution(generator) << ' ';
        }

        std::cout << '\n' << '\n';
    }

    return 0;
}
