#pragma once

#include <Eigen/Dense>
#include <memory>

#include "hypercube.h"


class Agent;

class FitnessFunction
{
protected:
    double dim;
    bool use_shared_function;
    double b_coef;
    double d_coef;

public:
    FitnessFunction(double dim, bool use_shared_function = true)
        :
        dim(dim),
        use_shared_function(use_shared_function),
        b_coef(0),
        d_coef(0)
    {}

    virtual double fitness(const Eigen::VectorXd &X) = 0;
    virtual Hypercube getBoundHypercube() = 0;
    virtual bool acceptable(const Eigen::VectorXd &X);
    virtual double volume();

    double calc(size_t index_in_model, const std::vector<std::shared_ptr<Agent>>& agents);
    double calc(Eigen::VectorXd X, size_t index_in_model, const std::vector<std::shared_ptr<Agent>>& agents);
    double sharing_function(Eigen::VectorXd X, size_t index_in_model, const std::vector<std::shared_ptr<Agent>>& agents);

    void setBCoef(double b_coef);
    void setDCoef(double d_coef);
};