#pragma once

#include "chicken.h"

class ChickenSwarm;
class Rootster;

class Hen : public Chicken
{
private:
    size_t parent_rootster_index;
    size_t hen_index;

    double calcC1(const std::shared_ptr<Rootster> &group_rootster, const std::vector<std::shared_ptr<AgentT<ChickenSwarm>>> &agents);
    double calcC2(const std::shared_ptr<Agent> &random_chicken, const std::vector<std::shared_ptr<AgentT<ChickenSwarm>>> &agents);

public:
    Hen(const Eigen::VectorXd &X, const std::shared_ptr<FitnessFunction>& fitness_function, size_t agent_index, size_t hen_index) :
        Chicken(X, fitness_function, agent_index),
        hen_index(hen_index)
    {}

    std::string getAgentType();
    Eigen::VectorXd calcMove(size_t t, bool before_role_update_move);

    void setParentRootsterAgentIndex(size_t parent_rootster_index);
};
