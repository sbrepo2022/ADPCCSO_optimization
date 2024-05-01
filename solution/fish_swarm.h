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
    void initAgents(const std::vector<Eigen::VectorXd> &X);

public:
    FishSwarm(
        const std::shared_ptr<FitnessFunction> &fitness_function,
        size_t agents_number
    )
        :
        Swarm(fitness_function, agents_number)
    {
    }

    void startupAgentsInit(const std::vector<Eigen::VectorXd> &X);
    void doMove(size_t t);

    void printData(bool verbose = false);
};
