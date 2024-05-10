#include "fitness_functions/rastrigin.h"

namespace fitness_function
{

double Rastrigin::fitness(const Eigen::VectorXd &X)
{
    double sum = 0.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        sum += pow(X[i], 2) - 10 * cos(2 * M_PI * X[i]) + 10;
    }
    return sum;
}


Hypercube Rastrigin::getBoundHypercube()
{
    return Hypercube(-Eigen::VectorXd::Ones(this->dim) * this->side_len / 2, this->side_len);
}

}
