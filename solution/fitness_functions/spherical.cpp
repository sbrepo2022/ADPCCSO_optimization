#include "fitness_functions/spherical.h"


namespace fitness_function
{


double Spherical::fitness(const Eigen::VectorXd &X)
{
    double g1 = 0.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        g1 += pow(X[i], 2) - pow(this->radius, 2);
    }

    double g1_plus = g1 > 0.0 ? g1 : 0.0;
    return pow(g1_plus, 2);
}


Hypercube Spherical::getBoundHypercube()
{
    return Hypercube(-Eigen::VectorXd::Ones(this->dim) * this->radius * sqrt(this->dim), 2 * sqrt(this->dim) * this->radius);
}


bool Spherical::acceptable(const Eigen::VectorXd &X)
{
    return std::abs(this->fitness(X)) < 1e-9;
}

}
