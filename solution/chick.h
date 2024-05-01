#pragma once

#include "chicken.h"


class ChickenSwarm;


class Chick : public Chicken
{
private:
    size_t parent_hen_index;
    size_t chick_index;

    Eigen::VectorXd defaultMove(const std::shared_ptr<ChickenSwarm> &swarm_ptr, size_t t);
    Eigen::VectorXd beforeRoleUpdateMove(const std::shared_ptr<ChickenSwarm> &swarm_ptr, size_t t);

public:
    Chick(const Eigen::VectorXd &X, const std::shared_ptr<FitnessFunction>& fitness_function, size_t agent_index, size_t chick_index) :
        Chicken(X, fitness_function, agent_index),
        chick_index(chick_index)
    {}

    std::string getAgentType();
    Eigen::VectorXd calcMove(size_t t, bool before_role_update_move);

    void setParentHenAgentIndex(size_t parent_hen_index);
};
