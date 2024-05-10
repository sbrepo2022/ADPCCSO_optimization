#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <limits>
#include <stdexcept>
#include <functional>
#include <future>
#include <mutex>

#include "fitness_function.h"
#include "agent_t.h"


template<class SwarmT>
class Swarm : public std::enable_shared_from_this<SwarmT>
{
protected:
    std::shared_ptr<FitnessFunction> fitness_function;

    std::vector<std::shared_ptr<AgentT<SwarmT>>> all_agents;         // Sorted by value of fitness function
    std::vector<std::shared_ptr<Agent>> generic_agents;         // Sorted by value of fitness function

    Eigen::VectorXd optimal_X;
    double optimal_value;
    size_t optimal_agent_index;

    size_t num_threads;

    virtual void initAgents(const std::vector<Eigen::VectorXd> &X, const std::vector<AgentClass> &agent_classes) = 0;


    int updateFitnessValues()
    {
        double best_value = std::numeric_limits<double>::max();
        int best_agent_index = -1;
        std::vector<double> fitness_values(this->all_agents.size(), std::numeric_limits<double>::max());

        auto thread_task = [this, &fitness_values](size_t start_i, size_t end_i)
        {
            for (size_t i = start_i; i < end_i; i++)
            {
                fitness_values[i] = this->all_agents[i]->getFitness()->calc(
                    this->all_agents[i]->getAgentIndex(),
                    this->generic_agents
                );
            }
        };

        size_t items_per_thread = this->all_agents.size() / this->num_threads;
        size_t threads_with_inc_items = this->all_agents.size() % this->num_threads;

        std::vector<std::future<void>> futures;
        size_t last_i = 0;
        for (size_t t_i = 0; t_i < this->num_threads; t_i++)
        {
            size_t next_i = last_i + (t_i < threads_with_inc_items ? items_per_thread + 1 : items_per_thread);
            futures.push_back(
                std::async(
                    this->num_threads > 1 ? std::launch::async : std::launch::deferred,
                    thread_task,
                    last_i,
                    next_i
                )
            );
            last_i = next_i;
        }

        for (auto& fut : futures)
        {
            fut.get();
        }

        for (size_t i = 0; i < this->all_agents.size(); i++)
        {
            this->all_agents[i]->updateCachedFitnessValue(fitness_values[i]);
            if (fitness_values[i] < best_value)
            {
                best_value = fitness_values[i];
                best_agent_index = i;
            }
        }
        //std::cout << "Best index: " << best_agent_index << " Best value: " << best_value << std::endl << std::endl;
        if (best_agent_index < 0)
            throw std::domain_error("Cannot find best agent index");
        return best_agent_index;
    }


    static std::vector<std::shared_ptr<Agent>> toGenericAgentVector(const std::vector<std::shared_ptr<AgentT<SwarmT>>> &specific_agents)
    {
        return std::vector<std::shared_ptr<Agent>>(specific_agents.begin(), specific_agents.end());
    }


public:
    Swarm(const std::shared_ptr<FitnessFunction> &fitness_function, size_t  num_threads)
        :
        fitness_function(fitness_function),
        num_threads(num_threads)
    {
        optimal_value = std::numeric_limits<double>::max();
    }


    virtual ~Swarm() {}


    static std::vector<AgentClass> calcAgentClasses(const std::shared_ptr<FitnessFunction> &fitness_function, const std::vector<Eigen::VectorXd> &X)
    {
        std::vector<AgentClass> agent_classes;
        for (auto& x: X)
        {
            if (fitness_function->acceptable(x))
            {
                agent_classes.push_back(AgentClass::ACCEPTABLE);
            }
            else
            {
                agent_classes.push_back(AgentClass::UNACCEPTABLE);
            }
        }
        return agent_classes;
    }


    void startupAgentsInit(const std::vector<Eigen::VectorXd> &X)
    {
        auto agent_classes = calcAgentClasses(this->fitness_function, X);
        this->initAgents(X, agent_classes);
    }


    void doMove(const std::function<Eigen::VectorXd(size_t)> &calc_move)
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
            for (size_t i = 0; i < this->all_agents.size(); i++)
            {
                AgentClass old_class = this->all_agents[i]->getAgentClass();
                Eigen::VectorXd &X_val = this->all_agents[i]->getX();
                AgentClass new_class = this->fitness_function->acceptable(X_val) ? AgentClass::ACCEPTABLE : AgentClass::UNACCEPTABLE;
                if (
                    old_class == AgentClass::ACCEPTABLE && new_class == AgentClass::UNACCEPTABLE ||
                    old_class == AgentClass::UNACCEPTABLE && new_class == AgentClass::ACCEPTABLE
                )
                {
                    this->all_agents[i]->updateAgentClass(AgentClass::ALMOST_ACCEPTABLE);
                }
            }
            this->optimal_value = best_value;
            this->optimal_X = best_X;
            this->optimal_agent_index = best_agent_index;
        }
    }


    std::vector<std::shared_ptr<AgentT<SwarmT>>>& getAgents()
    {
        return this->all_agents;
    }


    std::vector<std::shared_ptr<Agent>>& getGenericAgents()
    {
        return this->generic_agents;
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
        std::cout << "Best agent index: " << this->getOptimalAgentIndex() << std::endl;
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
