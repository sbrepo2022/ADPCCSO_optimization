#include "fitness_functions/alpine_1.h"


namespace fitness_function
{

double Alpine1::fitness(const Eigen::VectorXd &X)
{
    double sum = 0.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        sum += std::abs(X[i] * sin(X[i]) + 0.1 * X[i]);
    }
    return sum;
}


Hypercube Alpine1::getBoundHypercube()
{
    return Hypercube(-Eigen::VectorXd::Ones(this->dim) * this->side_len / 2, this->side_len);
}

}
