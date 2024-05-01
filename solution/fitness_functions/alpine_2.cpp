#include "fitness_functions/alpine_2.h"


namespace fitness_function
{

double Alpine2::fitness(const Eigen::VectorXd &X)
{
    double sum = 1.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        sum *= sqrt(X[i])*sin(X[i]);
    }
    return sum;
}


Hypercube Alpine2::getBoundHypercube()
{
    return Hypercube(Eigen::VectorXd::Zero(this->dim), this->side_len);
}

}
