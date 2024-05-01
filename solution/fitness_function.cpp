#include "fitness_function.h"
#include "agent.h"

#include <cmath>


double FitnessFunction::calc(size_t index_in_model, const std::vector<std::shared_ptr<Agent>>& agents)
{
    double sum = this->fitness(agents[index_in_model]->getX());
    if (this->use_shared_function)
    {
        sum += this->sharing_function(agents[index_in_model]->getX(), index_in_model, agents);
    }
    return sum;
}


double FitnessFunction::calc(Eigen::VectorXd X, size_t index_in_model, const std::vector<std::shared_ptr<Agent>>& agents)
{
    double sum = this->fitness(X);
    if (this->use_shared_function)
    {
        sum += this->sharing_function(X, index_in_model, agents);
    }
    return sum;
}


double FitnessFunction::sharing_function(Eigen::VectorXd X, size_t index_in_model, const std::vector<std::shared_ptr<Agent>>& agents)
{
    double sum = 0;
    for (size_t i = 0; i < agents.size(); i++)
    {
        if (i != index_in_model)
        {
            double norm = (agents[i]->getX() - X).norm();
            if (norm < this->d_coef)
            {
                sum += pow((1 - norm/this->d_coef), this->b_coef);
            }
        }
    }

    return sum;
}


void FitnessFunction::setBCoef(double b_coef)
{
    this->b_coef = b_coef;
}

void FitnessFunction::setDCoef(double d_coef)
{
    this->d_coef = d_coef;
}
