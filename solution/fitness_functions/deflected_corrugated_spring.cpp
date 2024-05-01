#include "fitness_functions/deflected_corrugated_spring.h"

namespace fitness_function
{

double DeflectedCorrugatedSpring::fitness(const Eigen::VectorXd &X)
{
    constexpr double alpha = 5;
    constexpr double K = 5;
    constexpr double A = 5;

    double inner_sum = 0.0;
    for (size_t j = 0; j < X.size(); j++)
    {
        inner_sum += pow(X[j] - alpha, 2);
    }

    double sum = 0.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        sum += pow(X[i] - alpha, 2) - A * cos(K * sqrt(inner_sum));
    }
    return 0.1 * sum;
}


Hypercube DeflectedCorrugatedSpring::getBoundHypercube()
{
    return Hypercube(Eigen::VectorXd::Zero(this->dim), this->side_len);
}

}