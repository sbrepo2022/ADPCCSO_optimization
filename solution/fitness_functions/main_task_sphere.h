#pragma once

#include "fitness_function.h"


namespace fitness_function
{

class MainTaskSphere : public FitnessFunction
{
protected:
    double radius;

public:
    MainTaskSphere(double dim, bool use_shared_function = true, double radius = 2.0)
        : FitnessFunction(dim, use_shared_function),
        radius(radius)
        {}

    double fitness(const Eigen::VectorXd &X) override;
    Hypercube getBoundHypercube() override;
    bool acceptable(const Eigen::VectorXd &X) override;
    double volume() override;
};

}
