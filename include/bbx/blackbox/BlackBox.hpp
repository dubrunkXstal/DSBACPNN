#pragma once

#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <bbx/activationfunctions/ActivationFunctions.hpp>
#include <bbx/activationfunctions/AnyActivationFunction.hpp>
#include <bbx/blackbox/Block.hpp>
#include <bbx/lossfunctions/LossFunctions.hpp>
#include <bbx/types.hpp>

namespace bbx {

class BlackBox {
public:
    BlackBox(std::ifstream& settings);

    // BlackBox(std::initializer_list<BlockConfig> block_configs);

    Vector evaluate(const Vector& x) const;

    void tuning(const Vector& x, const Vector& y);

private:
    size_t blocks_cnt;
    std::vector<std::unique_ptr<Block> > blocks;
    LossFunction loss;
};


// Implementation

inline BlackBox::BlackBox(std::ifstream& settings) : blocks(std::vector<std::unique_ptr<Block> >()) {
    Index in_dim;
    Index out_dim;
    std::string activaton;
    std::string line;

    std::getline(settings, line);
    std::stringstream ss(line);
    ss >> blocks_cnt;

    for (int i = 0; i < blocks_cnt; ++i) {
        getline(settings, line);
        ss = std::stringstream(line);
        ss >> in_dim >> out_dim >> activaton;

        if (activaton == "sigmoid") {
            blocks.emplace_back(std::make_unique<Block>(in_dim, out_dim, Sigmoid()));
        } else if (activaton == "relu") {
            blocks.emplace_back(std::make_unique<Block>(in_dim, out_dim, Relu()));
        } else {
            throw std::runtime_error("Didn't found activaton function for the block.");
        }
    }
}

// inline BlackBox::BlackBox(std::initializer_list<BlockConfig> block_configs) {
//     for (const auto& block_config : block_configs) {
//         blocks.emplace_back(std::make_unique<Block>(block_config.input_dimension, block_config.output_dimension, std::forward(block_config.activation_function)));
//     }
// }

inline Vector BlackBox::evaluate(const Vector& x) const {
    Vector result = x;

    for (int i = 0; i < blocks_cnt; ++i) {
        result = blocks[i]->evaluate(result);
    }

    return result;
}

inline void BlackBox::tuning(const Vector& x, const Vector& y) {
    std::vector<std::unique_ptr<Vector> > remember_output;
    remember_output.emplace_back(std::make_unique<Vector>(blocks[0]->evaluate(x)));

    for (int i = 1; i < blocks_cnt; ++i) {
        remember_output.emplace_back(
            std::make_unique<Vector>(blocks[i]->evaluate(*remember_output[i - 1])));
    }

    RowVector u = loss.gradient(*(remember_output[blocks_cnt - 1]), y);
    RowVector u_next;

    for (int i = blocks_cnt - 1; i > 0; --i) {
        u_next = blocks[i]->propogateBack(*(remember_output[i - 1]), u);
        blocks[i]->gradientDescent(*(remember_output[i - 1]), u);
        u = u_next;
    }

    blocks[0]->gradientDescent(x, u);
}

}  // namespace bbx
