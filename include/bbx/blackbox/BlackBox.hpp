#pragma once

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <filesystem>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <bbx/activationfunctions/ActivationFunctions.hpp>
#include <bbx/activationfunctions/AnyActivationFunction.hpp>
#include <bbx/blackbox/Block.hpp>
#include <bbx/lossfunctions/LossFunctions.hpp>
#include <bbx/lossfunctions/AnyLossFunction.hpp>
#include <bbx/types.hpp>

namespace bbx {

class BlackBox {
public:
    BlackBox(std::ifstream& settings);

    BlackBox(std::initializer_list<BlockConfig> block_configs);

    void setLoss(AnyLoss loss_function)
    {
        loss = loss_function;
    }

    Vector evaluate(const Vector& x) const;

    void tuning(const Vector& x, const Vector& y);

    void tuning(const Matrix& x_batch, const Matrix& y_batch);

    void loadCSV(const std::filesystem::path& path, std::vector<int> target_cols_idx, char delimiter = ',', bool have_header = false, int batch_size = 64);

    size_t getBlocksCount() const
    {
        return blocks.size();
    }

private:
    std::vector<std::unique_ptr<Block> > blocks;
    AnyLoss loss;
};


// Implementation

inline BlackBox::BlackBox(std::ifstream& settings) : blocks(std::vector<std::unique_ptr<Block> >()), loss(L2NormSquared{}) {
    Index in_dim;
    Index out_dim;
    std::string activaton;
    std::string line;

    std::getline(settings, line);
    std::stringstream ss(line);
    size_t blocks_cnt;
    ss >> blocks_cnt;

    for (int i = 0; i < blocks_cnt; ++i) {
        getline(settings, line);
        ss = std::stringstream(line);
        ss >> in_dim >> out_dim >> activaton;

        if (activaton == "sigmoid") {
            blocks.emplace_back(std::make_unique<Block>(in_dim, out_dim, Sigmoid{}));
        } else if (activaton == "relu") {
            blocks.emplace_back(std::make_unique<Block>(in_dim, out_dim, Relu{}));
        } else {
            throw std::runtime_error("BlackBox::BlackBox(): Specified activaton function for the block is unknown.");
        }
    }
}

inline BlackBox::BlackBox(std::initializer_list<BlockConfig> block_configs) : loss(L2NormSquared{}) {
    for (const auto& config : block_configs) {
        blocks.emplace_back(
            std::make_unique<Block>(
                config.input_dimension,
                config.output_dimension,
                config.activation_function
            )
        );
    }
}

inline Vector BlackBox::evaluate(const Vector& x) const {
    Vector result = x;

    for (int i = 0; i < getBlocksCount(); ++i) {
        result = blocks[i]->evaluate(result);
    }

    return result;
}

inline void BlackBox::tuning(const Vector& x, const Vector& y) {
    std::vector<std::unique_ptr<Vector> > remember_output;
    remember_output.emplace_back(std::make_unique<Vector>(blocks[0]->evaluate(x)));

    for (int i = 1; i < getBlocksCount(); ++i) {
        remember_output.emplace_back(
            std::make_unique<Vector>(blocks[i]->evaluate(*remember_output[i - 1])));
    }

    RowVector u = loss.gradient(*(remember_output[getBlocksCount() - 1]), y);
    RowVector u_next;

    for (int i = getBlocksCount() - 1; i > 0; --i) {
        u_next = blocks[i]->propogateBack(*(remember_output[i - 1]), u);
        blocks[i]->gradientDescent(*(remember_output[i - 1]), u);
        u = u_next;
    }

    blocks[0]->gradientDescent(x, u);
}

inline void BlackBox::tuning(const Matrix& x_batch, const Matrix& y_batch) {
    std::vector<std::unique_ptr<Matrix> > remember_output;

    remember_output.emplace_back(std::make_unique<Matrix>(blocks[0]->evaluate(x_batch)));
    for (int i = 1; i < getBlocksCount(); ++i) {
        remember_output.emplace_back(
            std::make_unique<Matrix>(blocks[i]->evaluate(*remember_output[i - 1])));
    }

    Matrix u = Matrix::Zero(y_batch.cols(), y_batch.rows());
    for (Index j = 0; j < u.rows(); ++j) {
        u.row(j) = loss.gradient((*(remember_output[getBlocksCount() - 1])).col(j), y_batch.col(j));
    }
    Matrix u_next;

    for (int i = getBlocksCount() - 1; i > 0; --i) {
        u_next = blocks[i]->propogateBack(*(remember_output[i - 1]), u);
        blocks[i]->gradientDescent(*(remember_output[i - 1]), u);
        u = u_next;
    }

    blocks[0]->gradientDescent(x_batch, u);
}

inline void BlackBox::loadCSV(const std::filesystem::path& path, std::vector<int> target_cols_idx, char delimiter, bool have_header, int batch_size)
{
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("BlackBox::loadCSV(): specified file not found.");
    }
    if (path.extension() != ".csv") {
        throw std::runtime_error("BlackBox::loadCSV(): provided file is not in CSV format.");
    }
    if (target_cols_idx.size() == 0) {
        throw std::runtime_error("BlackBox::loadCSV(): no target columns specified.");
    }

    std::ifstream csv(path);
    if (!csv.is_open()){
        throw std::runtime_error("BlackBox::loadCSV(): cannot open the file.");
    }

    std::string line;
    std::stringstream ss;
    std::getline(csv, line);

    int cols_cnt = std::count(line.begin(), line.end(), delimiter) + 1;

    for (int& i : target_cols_idx) {
        if (i >= cols_cnt) {
            throw std::runtime_error("BlackBox::loadCSV(): some specified target column index exceeds total amount of colums.");
        }
    }

    std::vector<std::vector<double> > x_accumuate;
    std::vector<std::vector<double> > y_accumuate;
    x_accumuate.reserve(batch_size); 
    y_accumuate.reserve(batch_size); 

    std::string value_str;

    std::vector<int> not_target_cols_idx;
    std::vector<double> vector_line(cols_cnt);

    for (int i = 0; i < cols_cnt; ++i) {
        if (std::find(target_cols_idx.begin(), target_cols_idx.end(), i) == target_cols_idx.end()) {
            not_target_cols_idx.emplace_back(i);
        }
    }

    int current_batch_size = 0;

    if (!have_header) {
        ss = std::stringstream(line);
        y_accumuate.emplace_back();
        x_accumuate.emplace_back();
        ++current_batch_size;

        for (int i = 0; i < cols_cnt; ++i) {
            std::getline(ss, value_str, delimiter);
            vector_line[i] = std::stod(value_str);
        }
        
        for (int& i : target_cols_idx) {
            y_accumuate.back().emplace_back(vector_line[i]);
        }

        for (int& i : not_target_cols_idx) {
            x_accumuate.back().emplace_back(vector_line[i]);
        }
    }

    while (std::getline(csv, line)) {
        ss = std::stringstream(line);
        y_accumuate.emplace_back();
        x_accumuate.emplace_back();
        ++current_batch_size;

        for (int i = 0; i < cols_cnt; ++i) {
            std::getline(ss, value_str, delimiter);
            vector_line[i] = std::stod(value_str);
        }
        
        for (int& i : target_cols_idx) {
            y_accumuate.back().emplace_back(vector_line[i]);
        }

        for (int& i : not_target_cols_idx) {
            x_accumuate.back().emplace_back(vector_line[i]);
        }

        if (current_batch_size == batch_size) {
            Matrix x_batch = Matrix::Zero(not_target_cols_idx.size(), batch_size);
            Matrix y_batch = Matrix::Zero(target_cols_idx.size(), batch_size);

            for (Index i = 0; i < batch_size; ++i) {
                x_batch.col(i) = Eigen::Map<Vector>(x_accumuate[i].data(), not_target_cols_idx.size());
            }
            
            for (Index i = 0; i < batch_size; ++i) {
                y_batch.col(i) = Eigen::Map<Vector>(y_accumuate[i].data(), target_cols_idx.size());
            }

            tuning(x_batch, y_batch);
            current_batch_size = 0;

            x_accumuate.clear();
            y_accumuate.clear();
        }
    }

    if (current_batch_size != 0) {
        Matrix x_batch = Matrix::Zero(not_target_cols_idx.size(), current_batch_size);
        Matrix y_batch = Matrix::Zero(target_cols_idx.size(), current_batch_size);

        for (Index i = 0; i < current_batch_size; ++i) {
            x_batch.col(i) = Eigen::Map<Vector>(x_accumuate[i].data(), not_target_cols_idx.size());
        }
        
        for (Index i = 0; i < current_batch_size; ++i) {
            y_batch.col(i) = Eigen::Map<Vector>(y_accumuate[i].data(), target_cols_idx.size());
        }

        tuning(x_batch, y_batch);
    }
    
}

}  // namespace bbx
