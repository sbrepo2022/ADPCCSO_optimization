#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <algorithm>

#include "fitness_function.h"
#include "hypercube.h"
#include "helpers.h"


class AgentInitializer
{
private:
    static void recursiveFill(std::vector<Eigen::VectorXd>& X, const Eigen::VectorXd& base_point, const Eigen::VectorXd& offsets, size_t num_agents_per_dim, size_t dim, Eigen::VectorXd current_point);

public:
    static std::vector<Eigen::VectorXd> hypercubeInitializer(const Hypercube &hypercube, size_t num_agents_per_dim, const std::shared_ptr<FitnessFunction>& fitness_function);
    static std::vector<Eigen::VectorXd> hypercubeUniformInitializer(const Hypercube &hypercube, size_t num_agents);
};
