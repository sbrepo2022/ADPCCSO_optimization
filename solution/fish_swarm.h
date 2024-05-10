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

    void initAgents(const std::vector<Eigen::VectorXd> &X, const std::vector<AgentClass> &agent_classes);

public:
    FishSwarm(const std::shared_ptr<FitnessFunction> &fitness_function, double step, double visual, size_t num_threads)
        :
        Swarm(fitness_function, num_threads),
        step(step),
        visual(visual)
    {
    }

    void printData(bool verbose = false);
};
