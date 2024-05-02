#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <limits>
#include <stdexcept>
#include <functional>

#include "fitness_function.h"
#include "agent_t.h"


template<class SwarmT>
class Swarm : public std::enable_shared_from_this<SwarmT>
{
protected:
    std::shared_ptr<FitnessFunction> fitness_function;

    std::vector<std::shared_ptr<AgentT<SwarmT>>> all_agents;         // Sorted by value of fitness function

    Eigen::VectorXd optimal_X;
    double optimal_value;
    size_t optimal_agent_index;

    virtual void initAgents(const std::vector<Eigen::VectorXd> &X) = 0;


    int updateFitnessValues()
    {
        double best_value = std::numeric_limits<double>::max();
        int best_agent_index = -1;
        for (size_t i = 0; i < this->all_agents.size(); i++)
        {
            double fitness_value = this->all_agents[i]->getFitness()->calc(
                this->all_agents[i]->getAgentIndex(),
                Swarm<SwarmT>::toGenericAgentVector(this->all_agents)
            );
            this->all_agents[i]->updateCachedFitnessValue(fitness_value);
            if (fitness_value < best_value)
            {
                best_value = fitness_value;
                best_agent_index = i;
            }
        }
        if (best_agent_index < 0)
            throw std::domain_error("Cannot find best agent index");
        return best_agent_index;
    }

public:
    static std::vector<std::shared_ptr<Agent>> toGenericAgentVector(const std::vector<std::shared_ptr<AgentT<SwarmT>>> &specific_agents)
    {
        return std::vector<std::shared_ptr<Agent>>(specific_agents.begin(), specific_agents.end());
    }


    Swarm(const std::shared_ptr<FitnessFunction> &fitness_function)
        :
        fitness_function(fitness_function)
    {
        optimal_value = std::numeric_limits<double>::max();
    }

    virtual ~Swarm() {}

    void startupAgentsInit(const std::vector<Eigen::VectorXd> &X)
    {
        this->initAgents(X);
    }


    void doMove(std::function<Eigen::VectorXd(size_t)> calc_move)
    {
        std::vector<Eigen::VectorXd> old_X;
        std::vector<Eigen::VectorXd> new_X;
        std::vector<double> old_fitness_values;

        for (auto& agent : this->all_agents)
        {
            old_X.push_back(agent->getX());
            old_fitness_values.push_back(agent->getCachedFitnessValue());
        }
        for (size_t i = 0; i < this->all_agents.size(); i++)
        {
            new_X.push_back(calc_move(i));
        }
        for (size_t i = 0; i < this->all_agents.size(); i++)
        {
            this->all_agents[i]->updateX(new_X[i]);
        }
        int best_agent_index = this->updateFitnessValues();
        double best_value = std::numeric_limits<double>::max();
        Eigen::VectorXd best_X;
        if (best_agent_index >= 0)
        {
            best_value = this->all_agents[best_agent_index]->getCachedFitnessValue();
            best_X = this->all_agents[best_agent_index]->getX();
        }

        // Undo if new best value worse than previous
        if (best_value > this->optimal_value)
        {
            for (size_t i = 0; i < this->all_agents.size(); i++)
            {
                this->all_agents[i]->updateX(old_X[i]);
                this->all_agents[i]->updateCachedFitnessValue(old_fitness_values[i]);
            }
        }
        else
        {
            this->optimal_value = best_value;
            this->optimal_X = best_X;
            this->optimal_agent_index = best_agent_index;
        }
    }


    std::vector<std::shared_ptr<AgentT<SwarmT>>>& getAgents()
    {
        return this->all_agents;
    }


    double getOptimalValue()
    {
        return this->optimal_value;
    }


    Eigen::VectorXd getOptimalX()
    {
        return this->optimal_X;
    }


    size_t getOptimalAgentIndex()
    {
        return this->optimal_agent_index;
    }


    virtual void printData(bool verbose = false)
    {
        std::cout << "\n";
        std::cout << "Best fitness value: " << this->getOptimalValue() << std::endl;
        std::cout << "Best position: [ " << this->getOptimalX().transpose() << " ]" << std::endl;

        if (verbose)
        {
            for (auto& agent : this->getAgents())
            {
                std::cout << "Agent [" << agent->getAgentIndex()
                        <<  "] in position: [ " << agent->getX().transpose()
                        << " ] with fitness value: " << agent->getCachedFitnessValue()
                        << std::endl;
            }
        }
        std::cout << "\n";
    }
};
