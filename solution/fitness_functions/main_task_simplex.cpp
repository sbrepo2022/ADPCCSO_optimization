#include "fitness_functions/main_task_simplex.h"


namespace fitness_function
{


double MainTaskSimplex::fitness(const Eigen::VectorXd &X)
{
    double sum = 0.0;
    double g = 0.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        g = -X[i];
        sum += pow(g > 0.0 ? g : 0.0, 2);
    }

    double sum_x = 0.0;
    for (int j = 0; j < X.size(); j++)
    {
        sum_x += X[j];
    }
    g = -h + sum_x;
    sum += pow(g > 0.0 ? g : 0.0, 2);

    return sum;
}


Hypercube MainTaskSimplex::getBoundHypercube()
{
    return Hypercube(-Eigen::VectorXd::Ones(this->dim) * this->h, 3 * this->h);
}


bool MainTaskSimplex::acceptable(const Eigen::VectorXd &X)
{
    return std::abs(this->fitness(X)) < 1e-9;
}


double MainTaskSimplex::volume()
{
    return 1./2. * pow(this->h, this->dim);
}

}
