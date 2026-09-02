#pragma once

#include <bbx/types.hpp>

namespace bbx {

class L2NormSquared {
   public:
    double distance(const Vector& z, const Vector& y) const
    {
        return pow((z - y).norm(), 2);
    }

    RowVector gradient(const Vector& z, const Vector& y) const;
};

// Implementation

inline RowVector L2NormSquared::gradient(const Vector& z, const Vector& y) const
{
    RowVector result(z.rows());

    for (int i = 0; i < z.rows(); ++i) {
        result[i] = 2 * (z[i] - y[i]);
    }

    return result;
}

}  // namespace bbx
