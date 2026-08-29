#pragma once

#include <Eigen/Core>

#include <bbx/activationfunctions/ActivationFunctions.hpp>

namespace bbx {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Index = Eigen::Index;

struct BlockConfig {
    Index input_dimension;
    Index output_dimension;
    CAny activation_function;
}

}  // namespace bbx
