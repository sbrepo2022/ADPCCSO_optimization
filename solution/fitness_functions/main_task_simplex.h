#pragma once

#include "fitness_function.h"

// TODO: Simplex

namespace fitness_function
{

class MainTaskSimplex : public FitnessFunction
{
protected:
    double h;

public:
    MainTaskSimplex(double dim, bool use_shared_function = true, double h = 2.0)
        : FitnessFunction(dim, use_shared_function),
        h(h)
        {}

    double fitness(const Eigen::VectorXd &X) override;
    Hypercube getBoundHypercube() override;
    bool acceptable(const Eigen::VectorXd &X) override;
    double volume() override;
};

}
