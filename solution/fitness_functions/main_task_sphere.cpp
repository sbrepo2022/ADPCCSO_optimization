#include "fitness_functions/main_task_sphere.h"


namespace fitness_function
{


double MainTaskSphere::fitness(const Eigen::VectorXd &X)
{
    double g = 0.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        g += pow(X[i], 2) - pow(this->radius, 2);
    }
    g = g > 0.0 ? g : 0;
    return pow(g, 2);
}


Hypercube MainTaskSphere::getBoundHypercube()
{
    return Hypercube(-Eigen::VectorXd::Ones(this->dim) * this->radius * sqrt(this->dim) * 1.5, 2 * sqrt(this->dim) * this->radius * 1.5);
}


bool MainTaskSphere::acceptable(const Eigen::VectorXd &X)
{
    return std::abs(this->fitness(X)) < 1e-9;
}


double MainTaskSphere::volume()
{
    double n = this->dim;
    double C = pow(M_PI, n/2) / tgamma(n/2 + 1);
    return C * pow(this->radius, n);
}

}
