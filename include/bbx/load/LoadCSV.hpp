#pragma once

#include <bbx/blackbox/BlackBox.hpp>

namespace bbx {

inline void BlackBox::loadTrainCSV(const std::filesystem::path& path, std::vector<int> target_cols_idx,
                                   char delimiter, bool have_header, int batch_size)
{
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("BlackBox::loadTrainCSV(): specified file not found.");
    }
    if (path.extension() != ".csv") {
        throw std::runtime_error("BlackBox::loadTrainCSV(): provided file is not in CSV format.");
    }
    if (target_cols_idx.size() == 0) {
        throw std::runtime_error("BlackBox::loadTrainCSV(): no target columns specified.");
    }

    std::ifstream csv(path);
    if (!csv.is_open()) {
        throw std::runtime_error("BlackBox::loadTrainCSV(): cannot open the file.");
    }

    std::string line;
    std::stringstream ss;
    std::getline(csv, line);

    int cols_cnt = std::count(line.begin(), line.end(), delimiter) + 1;

    for (int& i : target_cols_idx) {
        if (i >= cols_cnt) {
            throw std::runtime_error(
                "BlackBox::loadTrainCSV(): some specified target column index exceeds total amount of "
                "colums.");
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

    if (have_header) {
        std::getline(csv, line);
    }

    do {
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

        if (current_batch_size >= batch_size) {
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
    } while (std::getline(csv, line));

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

inline double BlackBox::loadTestCSV(const std::filesystem::path& path, std::vector<int> target_cols_idx,
                                    std::function<bool(const Vector&, const Vector&)> is_correct,
                                    char delimiter, bool have_header)
{
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("BlackBox::loadTestCSV(): specified file not found.");
    }
    if (path.extension() != ".csv") {
        throw std::runtime_error("BlackBox::loadTestCSV(): provided file is not in CSV format.");
    }
    if (target_cols_idx.size() == 0) {
        throw std::runtime_error("BlackBox::loadTestCSV(): no target columns specified.");
    }

    std::ifstream csv(path);
    if (!csv.is_open()) {
        throw std::runtime_error("BlackBox::loadTestCSV(): cannot open the file.");
    }

    std::string line;
    std::string value_str;
    std::stringstream ss;
    std::getline(csv, line);

    int cols_cnt = std::count(line.begin(), line.end(), delimiter) + 1;

    for (int& i : target_cols_idx) {
        if (i >= cols_cnt) {
            throw std::runtime_error(
                "BlackBox::loadTestCSV(): some specified target column index exceeds total amount of "
                "colums.");
        }
    }

    std::vector<int> not_target_cols_idx;
    std::vector<double> vector_line(cols_cnt);

    for (int i = 0; i < cols_cnt; ++i) {
        if (std::find(target_cols_idx.begin(), target_cols_idx.end(), i) == target_cols_idx.end()) {
            not_target_cols_idx.emplace_back(i);
        }
    }

    int total_cnt = 0;
    int success_cnt = 0;

    std::vector<double> x;
    std::vector<double> y;

    if (have_header) {
        std::getline(csv, line);
    }

    do {
        ss = std::stringstream(line);

        for (int i = 0; i < cols_cnt; ++i) {
            std::getline(ss, value_str, delimiter);
            vector_line[i] = std::stod(value_str);
        }

        for (int& i : target_cols_idx) {
            y.emplace_back(vector_line[i]);
        }

        for (int& i : not_target_cols_idx) {
            x.emplace_back(vector_line[i]);
        }

        ++total_cnt;
        if (is_correct(evaluate(Eigen::Map<Vector>(x.data(), x.size())),
                       Eigen::Map<Vector>(y.data(), y.size()))) {
            ++success_cnt;
        }

        x.clear();
        y.clear();
    } while (std::getline(csv, line));

    return total_cnt > 0 ? (double)success_cnt / total_cnt : 0;
}

}  // namespace bbx
