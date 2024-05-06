#include "hen.h"
#include "chicken_swarm.h"
#include "rootster.h"

std::string Hen::getAgentType()
{
    return std::string("Hen");
}


double Hen::calcC1(const std::shared_ptr<Rootster> &group_rootster, const std::vector<std::shared_ptr<AgentT<ChickenSwarm>>> &agents)
{
    double f_i = this->getCachedFitnessValue();
    double f_r1 = group_rootster->getCachedFitnessValue();
    if (f_i < f_r1) std::swap(f_i, f_r1);

    return exp((f_i - f_r1) / (fabs(f_i) + std::numeric_limits<double>::min()));
}


double Hen::calcC2(const std::shared_ptr<Agent> &random_chicken, const std::vector<std::shared_ptr<AgentT<ChickenSwarm>>> &agents)
{
    double f_i = this->getCachedFitnessValue();
    double f_r2 = random_chicken->getCachedFitnessValue();
    if (f_i < f_r2) std::swap(f_i, f_r2);

    //std::cout << "f_i: " << f_i << " f_r2: " << f_r2 << std::endl;

    return exp(f_r2 - f_i);
}


Eigen::VectorXd Hen::calcMove(size_t t, bool before_role_update_move)
{
    if (auto swarm_ptr = this->getSwarm().lock()) {
        decltype(auto) agents = swarm_ptr->getAgents();
        decltype(auto) rootsters = swarm_ptr->getRootsters();
        decltype(auto) group_rootster = rootsters[this->parent_rootster_index];

        size_t r = group_rootster->getAgentIndex();

        size_t i = this->getAgentIndex();
        size_t k = rand() % rootsters.size();
        while (k == i || k == r)
        {
            k++;
            if (k == rootsters.size()) k = 0;
        }

        decltype(auto) random_chicken = agents[k];

        double c1 = this->calcC1(group_rootster, agents);
        double c2 = this->calcC2(random_chicken, agents);
        //std::cout << "c1: " << c1 << " c2: " << c2 << std::endl;

        return this->calcLearningFactor(t) * this->getX()
                + c1 * math_helpers::uniform() * (group_rootster->getX() - this->getX())
                + c2 * math_helpers::uniform() * (swarm_ptr->getOptimalX() - this->getX());
    }
    return this->getX();
}


void Hen::setParentRootsterAgentIndex(size_t parent_rootster_index)
{
    this->parent_rootster_index = parent_rootster_index;
}
