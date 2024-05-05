#include "agent.h"
#include "chicken_swarm.h"


std::string Agent::getAgentType()
{
    return "Agent";
}


Eigen::VectorXd Agent::calcMove(size_t t, bool before_role_update_move)
{
    return this->X;
}


void Agent::updateX(const Eigen::VectorXd &X)
{
    this->X = X;
}


void Agent::updateCachedFitnessValue(double cached_fitness_value)
{
    this->cached_fitness_value = cached_fitness_value;
}


double Agent::getCachedFitnessValue()
{
    return this->cached_fitness_value;
}


Eigen::VectorXd& Agent::getX()
{
    return this->X;
}


std::shared_ptr<FitnessFunction>& Agent::getFitness()
{
    return this->fitness_function;
}


Agent::~Agent()
{

}


size_t Agent::getAgentIndex()
{
    return this->agent_index;
}


void Agent::updateAgentIndex(size_t agent_index)
{
    this->agent_index = agent_index;
}


AgentClass Agent::getAgentClass()
{
    return this->agent_class;
}

void Agent::updateAgentClass(AgentClass agent_class)
{
    this->agent_class = agent_class;
}
