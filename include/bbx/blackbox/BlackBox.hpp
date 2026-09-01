#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <bbx/activationfunctions/ActivationFunctions.hpp>
#include <bbx/activationfunctions/AnyActivationFunction.hpp>
#include <bbx/blackbox/Block.hpp>
#include <bbx/lossfunctions/AnyLossFunction.hpp>
#include <bbx/lossfunctions/LossFunctions.hpp>
#include <bbx/types.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace bbx {

class BlackBox {
   public:
    BlackBox(std::ifstream& settings);

    BlackBox(std::initializer_list<BlockConfig> block_configs);

    Vector evaluate(const Vector& x) const;

    void tuning(const Vector& x, const Vector& y);

    void tuning(const Matrix& x_batch, const Matrix& y_batch);

    void loadTrainCSV(const std::filesystem::path& path, std::vector<int> target_cols_idx,
                      char delimiter = ',', bool have_header = false, int batch_size = 32);

    double loadTestCSV(const std::filesystem::path& path, std::vector<int> target_cols_idx,
                       std::function<bool(const Vector&, const Vector&)> is_correct, char delimiter = ',',
                       bool have_header = false);

    size_t getBlocksCount() const { return blocks_.size(); }

    void setLoss(AnyLoss loss_function) { loss_ = loss_function; }

    void setOptimizer(const AnyOptimizer& optimizer, std::vector<int> blocks_idx) {
        for (int& id : blocks_idx) {
            blocks_[id]->setOptimizer(optimizer);
        }
    }

   private:
    std::vector<std::unique_ptr<Block> > blocks_;
    AnyLoss loss_;
};

// Implementation

inline BlackBox::BlackBox(std::ifstream& settings)
    : blocks_(std::vector<std::unique_ptr<Block> >()), loss_(L2NormSquared{}) {
    Index in_dim;
    Index out_dim;
    std::string activation;
    std::string line;

    std::getline(settings, line);
    std::stringstream ss(line);
    size_t blocks_cnt;
    ss >> blocks_cnt;

    for (int i = 0; i < blocks_cnt; ++i) {
        getline(settings, line);
        ss = std::stringstream(line);
        ss >> in_dim >> out_dim >> activation;

        if (activation == "sigmoid") {
            blocks_.emplace_back(std::make_unique<Block>(in_dim, out_dim, Sigmoid{}));
        } else if (activation == "relu") {
            blocks_.emplace_back(std::make_unique<Block>(in_dim, out_dim, Relu{}));
        } else {
            throw std::runtime_error(
                "BlackBox::BlackBox(): Specified activation function for the block is unknown.");
        }
    }
}

inline BlackBox::BlackBox(std::initializer_list<BlockConfig> block_configs) : loss_(L2NormSquared{}) {
    for (const auto& config : block_configs) {
        blocks_.emplace_back(std::make_unique<Block>(config.input_dimension, config.output_dimension,
                                                     config.activation_function, config.optimizer));
    }
}

inline Vector BlackBox::evaluate(const Vector& x) const {
    Vector result = x;

    for (int i = 0; i < getBlocksCount(); ++i) {
        result = blocks_[i]->evaluate(result);
    }

    return result;
}

inline void BlackBox::tuning(const Vector& x, const Vector& y) {
    std::vector<std::unique_ptr<Vector> > remember_output;
    remember_output.emplace_back(std::make_unique<Vector>(blocks_[0]->evaluate(x)));

    for (int i = 1; i < getBlocksCount(); ++i) {
        remember_output.emplace_back(std::make_unique<Vector>(blocks_[i]->evaluate(*remember_output[i - 1])));
    }

    RowVector u = loss_.gradient(*(remember_output[getBlocksCount() - 1]), y);
    RowVector u_next;

    for (int i = getBlocksCount() - 1; i > 0; --i) {
        u_next = blocks_[i]->propogateBack(*(remember_output[i - 1]), u);
        blocks_[i]->gradientDescent(*(remember_output[i - 1]), u);
        u = u_next;
    }

    blocks_[0]->gradientDescent(x, u);
}

inline void BlackBox::tuning(const Matrix& x_batch, const Matrix& y_batch) {
    std::vector<std::unique_ptr<Matrix> > remember_output;

    remember_output.emplace_back(std::make_unique<Matrix>(blocks_[0]->evaluate(x_batch)));
    for (int i = 1; i < getBlocksCount(); ++i) {
        remember_output.emplace_back(std::make_unique<Matrix>(blocks_[i]->evaluate(*remember_output[i - 1])));
    }

    Matrix u = Matrix::Zero(y_batch.cols(), y_batch.rows());
    for (Index j = 0; j < u.rows(); ++j) {
        u.row(j) = loss_.gradient((*(remember_output[getBlocksCount() - 1])).col(j), y_batch.col(j));
    }
    Matrix u_next;

    for (int i = getBlocksCount() - 1; i > 0; --i) {
        u_next = blocks_[i]->propogateBack(*(remember_output[i - 1]), u);
        blocks_[i]->gradientDescent(*(remember_output[i - 1]), u);
        u = u_next;
    }

    blocks_[0]->gradientDescent(x_batch, u);
}

}  // namespace bbx

#include <bbx/load/LoadCSV.hpp>
