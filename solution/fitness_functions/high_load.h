#pragma once

#include "fitness_function.h"


namespace fitness_function
{

class HighLoad : public FitnessFunction
{
protected:
    double radius;

public:
    HighLoad(double dim, bool use_shared_function = false, double radius = 2.0)
        : FitnessFunction(dim, use_shared_function),
        radius(radius)
        {}

    double fitness(const Eigen::VectorXd &X) override;
    Hypercube getBoundHypercube() override;
};

}
