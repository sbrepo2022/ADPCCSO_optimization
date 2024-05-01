#include "fitness_functions/spherical.h"


namespace fitness_function
{

double Spherical::fitness(const Eigen::VectorXd &X)
{
    double sum = 0.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        sum += pow(X[i] + 0.01, 2) - pow(this->radius, 2);
    }
    return std::abs(sum);
}


Hypercube Spherical::getBoundHypercube()
{
    return Hypercube(-Eigen::VectorXd::Ones(this->dim) * this->radius * sqrt(this->dim), 2 * sqrt(this->dim) * this->radius);
}

}
