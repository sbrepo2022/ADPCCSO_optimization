#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <string>
#include <cmath>

#include "fitness_function.h"
#include "helpers.h"
#include "agent.h"


template<class SwarmT>
class AgentT : public Agent
{
private:
    std::weak_ptr<SwarmT> swarm;

public:
    AgentT(const Eigen::VectorXd &X, const std::shared_ptr<FitnessFunction>& fitness_function, size_t agent_index)
        :
        Agent(X, fitness_function, agent_index)
    {}

    void attachToSwarm(const std::shared_ptr<SwarmT> &swarm)
    {
        this->swarm = swarm;
    }


    std::weak_ptr<SwarmT>& getSwarm()
    {
        return this->swarm;
    }

    virtual ~AgentT() {}
};
