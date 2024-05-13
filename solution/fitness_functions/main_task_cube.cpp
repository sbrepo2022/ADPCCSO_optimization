#include "fitness_functions/main_task_cube.h"


namespace fitness_function
{


double MainTaskCube::fitness(const Eigen::VectorXd &X)
{
    double sum = 0.0;
    double g = 0.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        g = X[i] - a;
        sum += pow(g > 0.0 ? g : 0.0, 2);
        g = -X[i];
        sum += pow(g > 0.0 ? g : 0.0, 2);
    }

    return sum;
}


Hypercube MainTaskCube::getBoundHypercube()
{
    return Hypercube(-Eigen::VectorXd::Ones(this->dim) * this->a, 3 * this->a);
}


bool MainTaskCube::acceptable(const Eigen::VectorXd &X)
{
    return std::abs(this->fitness(X)) < 1e-9;
}


double MainTaskCube::volume()
{
    return pow(this->a, this->dim);
}

}
