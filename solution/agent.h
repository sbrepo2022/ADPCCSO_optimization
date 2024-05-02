#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <string>
#include <cmath>

#include "fitness_function.h"
#include "helpers.h"


class Agent
{
protected:
    Eigen::VectorXd X;
    std::shared_ptr<FitnessFunction> fitness_function;
    size_t agent_index;

    double cached_fitness_value;

public:
    Agent(const Eigen::VectorXd &X, const std::shared_ptr<FitnessFunction>& fitness_function, size_t agent_index) :
        X(X),
        fitness_function(fitness_function),
        agent_index(agent_index)
    {}

    virtual std::string getAgentType();
    virtual Eigen::VectorXd calcMove(size_t t, bool before_role_update_move);

    void updateCachedFitnessValue(double cached_fitness_value);
    double getCachedFitnessValue();

    Eigen::VectorXd& getX();
    void updateX(const Eigen::VectorXd &X);
    std::shared_ptr<FitnessFunction>& getFitness();
    size_t getAgentIndex();
    void updateAgentIndex(size_t agent_index);

    virtual ~Agent();
};
