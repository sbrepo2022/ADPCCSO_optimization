#include "fish_swarm.h"
#include "fish.h"


void FishSwarm::initAgents(const std::vector<Eigen::VectorXd> &X, const std::vector<AgentClass> &agent_classes)
{
    this->all_agents.clear();

    std::vector<std::shared_ptr<Agent>> agents_tmp;
    for (int i = 0; i < X.size(); i++)
    {
        auto fish = std::make_shared<Fish>(X[i], this->fitness_function, this->all_agents.size(), this->step, this->visual);
        fish->attachToSwarm(this->shared_from_this());
        fish->updateAgentClass(agent_classes[i]);
        this->all_agents.push_back(fish);
    }

    this->generic_agents = Swarm::toGenericAgentVector(this->all_agents);

    size_t best_agent_index = this->updateFitnessValues();
    this->optimal_value = this->all_agents[best_agent_index]->getCachedFitnessValue();
    this->optimal_X = this->all_agents[best_agent_index]->getX();
}


void FishSwarm::printData(bool verbose)
{
    std::cout << "Fish swarm data:";
    Swarm::printData(verbose);
}
