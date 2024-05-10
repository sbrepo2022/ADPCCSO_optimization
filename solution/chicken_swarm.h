#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <limits>
#include <stdexcept>

#include "swarm.h"
#include "fitness_function.h"

class Rootster;
class Hen;
class Chick;


class ChickenSwarm : public Swarm<ChickenSwarm>
{
private:
    size_t rootsters_number;
    size_t hens_number;
    size_t chicks_number;

    size_t max_iterations;
    double learn_factor_min;
    double learn_factor_max;

    size_t num_threads;

    std::vector<std::shared_ptr<Rootster>> all_rootsters;
    std::vector<std::shared_ptr<Hen>> all_hens;
    std::vector<std::shared_ptr<Chick>> all_chicks;

    void initAgents(const std::vector<Eigen::VectorXd> &X, const std::vector<AgentClass> &agent_classes);

public:
    ChickenSwarm(
        const std::shared_ptr<FitnessFunction> &fitness_function,
        size_t rootsters_number,
        size_t hens_number,
        size_t chicks_number,
        size_t max_iterations,
        double learn_factor_min,
        double learn_factor_max,
        size_t num_threads
    )
        :
        Swarm(fitness_function, num_threads),
        rootsters_number(rootsters_number),
        hens_number(hens_number),
        chicks_number(chicks_number),
        max_iterations(max_iterations),
        learn_factor_min(learn_factor_min),
        learn_factor_max(learn_factor_max)
    {
    }

    void updateAgentsRoles();

    std::vector<std::shared_ptr<Rootster>>& getRootsters();
    std::vector<std::shared_ptr<Hen>>& getHens();
    std::vector<std::shared_ptr<Chick>>& getChick();

    size_t getMaxIterations();
    double getLearnFactorMin();
    double getLearnFactorMax();

    void printData(bool verbose = false);
};
