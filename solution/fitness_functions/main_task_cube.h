#pragma once

#include "fitness_function.h"

// TODO: Cube

namespace fitness_function
{

class MainTaskCube : public FitnessFunction
{
protected:
    double a;

public:
    MainTaskCube(double dim, bool use_shared_function = true, double a = 2.0)
        : FitnessFunction(dim, use_shared_function),
        a(a)
        {}

    double fitness(const Eigen::VectorXd &X) override;
    Hypercube getBoundHypercube() override;
    bool acceptable(const Eigen::VectorXd &X) override;
    double volume() override;
};

}
