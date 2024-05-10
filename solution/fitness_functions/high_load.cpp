#include "fitness_functions/high_load.h"

#include <chrono>
#include <thread>


namespace fitness_function
{


double HighLoad::fitness(const Eigen::VectorXd &X)
{
    std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    double sum = 0.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        sum += pow(X[i], 2);
    }
    return sum;
}


Hypercube HighLoad::getBoundHypercube()
{
    return Hypercube(-Eigen::VectorXd::Ones(this->dim) * this->radius * sqrt(this->dim), 2 * sqrt(this->dim) * this->radius);
}

}
