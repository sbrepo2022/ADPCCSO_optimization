#pragma once

#include "fitness_function.h"


namespace fitness_function
{

class Alpine1 : public FitnessFunction
{
protected:
    double side_len;

public:
    Alpine1(double dim, bool use_shared_function = false, double side_len = 20.0)
        : FitnessFunction(dim, false),
        side_len(side_len)
        {}

    double fitness(const Eigen::VectorXd &X) override;
    Hypercube getBoundHypercube() override;
};

}
