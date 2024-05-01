#pragma once

#include "fitness_function.h"


namespace fitness_function
{

class Spherical : public FitnessFunction
{
protected:
    double radius;

public:
    Spherical(double dim, bool use_shared_function = false, double radius = 5.0)
        : FitnessFunction(dim, use_shared_function),
        radius(radius)
        {}

    double fitness(const Eigen::VectorXd &X) override;
    Hypercube getBoundHypercube() override;
};

}
