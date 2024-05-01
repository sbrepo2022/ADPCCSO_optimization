#include "rootster.h"
#include "chicken_swarm.h"


std::string Rootster::getAgentType()
{
    return std::string("Rootster");
}


double Rootster::rand_std_dev(const std::shared_ptr<Rootster> &random_rootster, const std::vector<std::shared_ptr<AgentT<ChickenSwarm>>> &agents)
{
    double f_i = this->getCachedFitnessValue();
    double f_k = random_rootster->getCachedFitnessValue();

    if (f_i <= f_k)
    {
        return 1.0;
    }
    else
    {
        return exp((f_k - f_i) / (fabs(f_i) + std::numeric_limits<double>::min()));
    }
}


Eigen::VectorXd Rootster::calcMove(size_t t, bool before_role_update_move)
{
    if (auto swarm_ptr = this->getSwarm().lock()) {
        decltype(auto) agents = swarm_ptr->getAgents();
        decltype(auto) rootsters = swarm_ptr->getRootsters();

        size_t i = this->rootster_index;
        size_t k = rand() % rootsters.size();
        if (k == i) k++;
        if (k == rootsters.size()) k = 0;

        decltype(auto) random_rootster = rootsters[k];

        double std_dev = this->rand_std_dev(random_rootster, agents);
        return this->calcLearningFactor(t) * this->getX() * (1 + math_helpers::randn(std_dev));
    }
    return this->getX();
}