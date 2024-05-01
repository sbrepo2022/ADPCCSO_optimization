#include "chick.h"
#include "chicken_swarm.h"
#include "hen.h"


std::string Chick::getAgentType()
{
    return std::string("Chick");
}


Eigen::VectorXd Chick::defaultMove(const std::shared_ptr<ChickenSwarm> &swarm_ptr, size_t t)
{
    decltype(auto) mother_hen = swarm_ptr->getHens()[this->parent_hen_index];

    double fl = rand() % 2 == 0 ? 0 : 2;

    return this->calcLearningFactor(t) * this->getX()
            + fl * (mother_hen->getX() - this->getX())
            + fl * (swarm_ptr->getOptimalX() - this->getX());
}


Eigen::VectorXd Chick::beforeRoleUpdateMove(const std::shared_ptr<ChickenSwarm> &swarm_ptr, size_t t)
{
    Eigen::VectorXd opt_x = swarm_ptr->getOptimalX();
    size_t dim = opt_x.size();
    double opt_x_norm = opt_x.norm();
    Eigen::VectorXd lb = opt_x - Eigen::VectorXd::Ones(dim) * opt_x_norm * math_helpers::uniform();
    Eigen::VectorXd ub = opt_x + Eigen::VectorXd::Ones(dim) * opt_x_norm * math_helpers::uniform();

    return lb + (ub - lb) * math_helpers::uniform();
}


Eigen::VectorXd Chick::calcMove(size_t t, bool before_role_update_move)
{
    if (auto swarm_ptr = this->getSwarm().lock()) {
        if (before_role_update_move)
        {
            return this->beforeRoleUpdateMove(swarm_ptr, t);
        }

        return this->defaultMove(swarm_ptr, t);
    }
    return this->getX();
}


void Chick::setParentHenAgentIndex(size_t parent_hen_index)
{
    this->parent_hen_index = parent_hen_index;
}
