#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <limits>
#include <stdexcept>

#include "swarm.h"
#include "fitness_function.h"

class Fish;

class FishSwarm : public Swarm<FishSwarm>
{
private:
    double step;
    double visual;

    void initAgents(const std::vector<Eigen::VectorXd> &X);

public:
    FishSwarm(const std::shared_ptr<FitnessFunction> &fitness_function, double step, double visual)
        :
        Swarm(fitness_function),
        step(step),
        visual(visual)
    {
    }

    void printData(bool verbose = false);
};
