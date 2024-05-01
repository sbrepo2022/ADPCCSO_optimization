#pragma once

#include "agent_t.h"
#include "fish_swarm.h"
#include "helpers.h"

class Fish : public AgentT<FishSwarm>
{
private:
    double step;
    double visual;

    Eigen::VectorXd calcPreyBehavior(const std::shared_ptr<FishSwarm> &swarm_ptr, size_t t);
    Eigen::VectorXd calcSwarmingBehavior(const std::shared_ptr<FishSwarm> &swarm_ptr, size_t t, const std::vector<std::shared_ptr<AgentT<FishSwarm>>> &agents);
    Eigen::VectorXd calcFollowingBehavior(const std::shared_ptr<FishSwarm> &swarm_ptr, size_t t);

public:
    Fish(const Eigen::VectorXd &X, const std::shared_ptr<FitnessFunction>& fitness_function, size_t agent_index, double step, double visual)
        :
        AgentT(X, fitness_function, agent_index),
        step(step),
        visual(visual)
    {}

    virtual ~Fish();

    Eigen::VectorXd calcMove(size_t t, bool before_role_update_move);
};