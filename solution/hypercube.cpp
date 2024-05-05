#include "hypercube.h"

bool Hypercube::isXIn(const Eigen::VectorXd &X) const
{
    for (int i = 0; i < X.size(); i++)
    {
        if (X[i] < this->base_point[i] || X[i] > this->base_point[i] + this->len) return false;
    }
    return true;
}