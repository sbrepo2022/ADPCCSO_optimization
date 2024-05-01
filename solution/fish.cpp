#include "fish.h"

Fish::~Fish()
{

}


Eigen::VectorXd Fish::calcPreyBehavior(const std::shared_ptr<FishSwarm> &swarm_ptr, size_t t)
{
    size_t dim = this->getX().size();
    Eigen::VectorXd X_i = this->getX();
    Eigen::VectorXd X_j = X_i + Eigen::VectorXd::Ones(dim) * math_helpers::uniform() * this->visual;
    return X_i + math_helpers::uniform() * this->step * (X_j - X_i).normalized();
}


Eigen::VectorXd Fish::calcSwarmingBehavior(const std::shared_ptr<FishSwarm> &swarm_ptr, size_t t, const std::vector<std::shared_ptr<AgentT<FishSwarm>>> &agents)
{
    size_t dim = this->getX().size();
    Eigen::VectorXd X_i = this->getX();

    size_t nf = 0;
    Eigen::VectorXd X_c = Eigen::VectorXd::Zero(dim);

    for (const auto& agent : agents)
    {
        Eigen::VectorXd X_ci = agent->getX();
        if ((X_i - X_ci).squaredNorm() <= pow(this->visual, 2))
        {
            X_c += X_ci;
            nf++;
        }
    }
    if (nf > 0)
    {
        X_c = X_c / nf;
    }

    return X_i + math_helpers::uniform() * this->step * (X_c - X_i).normalized();
}


Eigen::VectorXd Fish::calcFollowingBehavior(const std::shared_ptr<FishSwarm> &swarm_ptr, size_t t)
{
    size_t dim = this->getX().size();
    Eigen::VectorXd X_i = this->getX();
    Eigen::VectorXd X_m = swarm_ptr->getOptimalX();
    return X_i + math_helpers::uniform() * this->step * (X_m - X_i).normalized();
}


Eigen::VectorXd Fish::calcMove(size_t t, bool before_role_update_move)
{
    if (auto swarm_ptr = this->getSwarm().lock()) {
        decltype(auto) agents = swarm_ptr->getAgents();

        Eigen::VectorXd prey_X = this->calcPreyBehavior(swarm_ptr, t);
        Eigen::VectorXd swarming_X = this->calcSwarmingBehavior(swarm_ptr, t, agents);
        Eigen::VectorXd following_X = this->calcFollowingBehavior(swarm_ptr, t);

        double prey_fitness_value = this->getFitness()->calc(
            prey_X,
            this->getAgentIndex(),
            Swarm<FishSwarm>::toGenericAgentVector(agents)
        );
        double swarming_fitness_value = this->getFitness()->calc(
            swarming_X,
            this->getAgentIndex(),
            Swarm<FishSwarm>::toGenericAgentVector(agents)
        );
        double following_fitness_value = this->getFitness()->calc(
            following_X,
            this->getAgentIndex(),
            Swarm<FishSwarm>::toGenericAgentVector(agents)
        );

        if (prey_fitness_value >= swarming_fitness_value && prey_fitness_value >= following_fitness_value)
        {
            return prey_X;
        }
        else if (swarming_fitness_value >= prey_fitness_value && swarming_fitness_value >= following_fitness_value)
        {
            return swarming_X;
        }
        else if (following_fitness_value >= prey_fitness_value && following_fitness_value >= swarming_fitness_value)
        {
            return following_X;
        }
    }
    return this->getX();
}
