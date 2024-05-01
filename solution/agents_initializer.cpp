#include "agents_initializer.h"


void AgentInitializer::recursiveFill(std::vector<Eigen::VectorXd>& X, const Eigen::VectorXd& base_point, const Eigen::VectorXd& offsets, size_t num_agents_per_dim, size_t dim, Eigen::VectorXd current_point)
    {
        if (dim == base_point.size())
        {
            X.push_back(base_point + current_point);
            return;
        }

        for (size_t i = 0; i < num_agents_per_dim; ++i)
        {
            current_point[dim] = offsets[i];
            recursiveFill(X, base_point, offsets, num_agents_per_dim, dim + 1, current_point);
        }
    }


std::vector<Eigen::VectorXd> AgentInitializer::hypercubeInitializer(const Hypercube &hypercube, size_t num_agents_per_dim, const std::shared_ptr<FitnessFunction>& fitness_function)
{
    std::vector<Eigen::VectorXd> X;
    size_t num_dim = hypercube.base_point.size();
    Eigen::VectorXd offsets = Eigen::VectorXd::LinSpaced(num_agents_per_dim, 0, hypercube.len - hypercube.len / num_agents_per_dim);

    recursiveFill(X, hypercube.base_point, offsets, num_agents_per_dim, 0, Eigen::VectorXd::Zero(num_dim));

    // X.erase(std::remove_if(X.begin(), X.end(), [&](auto& x) {
    //     return fitness_function->fitness(x) < 0;
    // }), X.end());

    return X;
}


std::vector<Eigen::VectorXd> AgentInitializer::hypercubeUniformInitializer(const Hypercube &hypercube, size_t num_agents)
{
    std::vector<Eigen::VectorXd> X;
    size_t num_dims = hypercube.base_point.size();

    for (size_t agent_i = 0; agent_i < num_agents; agent_i++)
    {
        Eigen::VectorXd new_X = Eigen::VectorXd::Zero(num_dims);
        for (size_t dim_i = 0; dim_i < num_dims; dim_i++)
        {
            new_X[dim_i] = math_helpers::uniform(hypercube.base_point[dim_i], hypercube.base_point[dim_i] + hypercube.len);
        }
        X.push_back(new_X);
    }

    return X;
}
