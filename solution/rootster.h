#pragma once

#include <cstdlib>
#include <limits>

#include "chicken.h"
#include "helpers.h"


class ChickenSwarm;

class Rootster : public Chicken
{
private:
    size_t rootster_index;

    double rand_std_dev(const std::shared_ptr<Rootster> &random_rootster, const std::vector<std::shared_ptr<AgentT<ChickenSwarm>>> &agents);

public:
    Rootster(const Eigen::VectorXd &X, const std::shared_ptr<FitnessFunction>& fitness_function, size_t agent_index_in_swarm, size_t rootster_index) :
        Chicken(X, fitness_function, agent_index_in_swarm),
        rootster_index(rootster_index)
    {}

    std::string getAgentType();
    Eigen::VectorXd calcMove(size_t t, bool before_role_update_move);
};
