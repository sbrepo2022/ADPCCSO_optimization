#pragma once

#include "agent_t.h"
#include "chicken_swarm.h"

class Chicken : public AgentT<ChickenSwarm>
{
public:
    Chicken(const Eigen::VectorXd &X, const std::shared_ptr<FitnessFunction>& fitness_function, size_t agent_index)
        :
        AgentT(X, fitness_function, agent_index)
    {}


    double calcLearningFactor(size_t t)
    {
        if (auto swarm_ptr = this->getSwarm().lock()) {
            double M = swarm_ptr->getMaxIterations();
            double lfac_min = swarm_ptr->getLearnFactorMax();
            double lfac_max = swarm_ptr->getLearnFactorMax();

            double a = static_cast<double>(t) * (log(lfac_max) - log(lfac_min)) / M - log(lfac_max);

            return exp(-a);
        }
        return 1;
    }


    virtual ~Chicken() {}
};
