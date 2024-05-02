#include <iostream>

#include "chicken_swarm.h"
#include "agent.h"
#include "rootster.h"
#include "hen.h"
#include "chick.h"


void ChickenSwarm::initAgents(const std::vector<Eigen::VectorXd> &X)
{
    // Инициализация, сортировка, группировка, назначение классов и иерархии.

    // Создание агентов
    this->all_agents.clear();

    std::vector<std::shared_ptr<Agent>> agents_tmp;
    for (auto& x : X)
    {
        agents_tmp.push_back(std::make_shared<Agent>(x, this->fitness_function, agents_tmp.size()));
    }

    // Вычисление значений фитнесс-функции
    for (size_t i = 0; i < agents_tmp.size(); i++)
    {
        agents_tmp[i]->updateCachedFitnessValue(agents_tmp[i]->getFitness()->calc(agents_tmp[i]->getAgentIndex(), agents_tmp));
    }

    // Сортировка агентов по возрастанию фитнесс функции
    std::sort(agents_tmp.begin(), agents_tmp.end(), [](const auto& agent1, const auto& agent2) {
        return agent1->getCachedFitnessValue() < agent2->getCachedFitnessValue();
    });

    // Обновление индексов агентов
    for (size_t i = 0; i < agents_tmp.size(); i++)
    {
        agents_tmp[i]->updateAgentIndex(i);
    }

    int agent_i = 0;

    // Создание петухов
    this->all_rootsters.clear();
    for (size_t rootster_i = 0; rootster_i < this->rootsters_number; rootster_i++)
    {
        auto rootster = std::make_shared<Rootster>(
            agents_tmp[agent_i]->getX(),
            agents_tmp[agent_i]->getFitness(),
            agents_tmp[agent_i]->getAgentIndex(),
            rootster_i
        );
        rootster->attachToSwarm(this->shared_from_this());
        this->all_rootsters.push_back(rootster);
        this->all_agents.push_back(rootster);
        agent_i++;
    }

    // Создание курочек
    this->all_hens.clear();
    for (size_t hen_i = 0; hen_i < this->hens_number; hen_i++)
    {
        auto hen = std::make_shared<Hen>(
            agents_tmp[agent_i]->getX(),
            agents_tmp[agent_i]->getFitness(),
            agents_tmp[agent_i]->getAgentIndex(),
            hen_i
        );
        hen->attachToSwarm(this->shared_from_this());
        this->all_hens.push_back(hen);
        this->all_agents.push_back(hen);
        agent_i++;
    }

    std::vector<size_t> rootster_hen_assignments = math_helpers::randomParentChildAssignments(this->hens_number, this->rootsters_number);
    for (size_t hen_i = 0; hen_i < this->hens_number; hen_i++)
    {
        this->all_hens[hen_i]->setParentRootsterAgentIndex(rootster_hen_assignments[hen_i]);
    }

    // Создание цыплят
    this->all_chicks.clear();
    for (size_t chick_i = 0; chick_i < this->chicks_number; chick_i++)
    {
        auto chick = std::make_shared<Chick>(
            agents_tmp[agent_i]->getX(),
            agents_tmp[agent_i]->getFitness(),
            agents_tmp[agent_i]->getAgentIndex(),
            chick_i
        );
        chick->attachToSwarm(this->shared_from_this());
        this->all_chicks.push_back(chick);
        this->all_agents.push_back(chick);
        agent_i++;
    }

    std::vector<size_t> hen_chicks_assignments = math_helpers::randomParentChildAssignments(this->chicks_number, this->hens_number);
    for (size_t chick_i = 0; chick_i < this->chicks_number; chick_i++)
    {
        this->all_chicks[chick_i]->setParentHenAgentIndex(hen_chicks_assignments[chick_i]);
    }

    this->generic_agents = Swarm::toGenericAgentVector(this->all_agents);

    size_t best_agent_index = this->updateFitnessValues();
    this->optimal_value = this->all_agents[best_agent_index]->getCachedFitnessValue();
    this->optimal_X = this->all_agents[best_agent_index]->getX();
}


void ChickenSwarm::updateAgentsRoles()
{
    std::vector<Eigen::VectorXd> X;
    for (auto& agent : this->all_agents)
    {
        X.push_back(agent->getX());
    }
    this->initAgents(X);
}


std::vector<std::shared_ptr<Rootster>>& ChickenSwarm::getRootsters()
{
    return this->all_rootsters;
}


std::vector<std::shared_ptr<Hen>>& ChickenSwarm::getHens()
{
    return this->all_hens;
}

std::vector<std::shared_ptr<Chick>>& ChickenSwarm::getChick()
{
    return this->all_chicks;
}


size_t ChickenSwarm::getMaxIterations()
{
    return this->max_iterations;
}


double ChickenSwarm::getLearnFactorMin()
{
    return this->learn_factor_min;
}


double ChickenSwarm::getLearnFactorMax()
{
    return this->learn_factor_max;
}


void ChickenSwarm::printData(bool verbose)
{
    std::cout << "Chicken swarm data:";
    Swarm::printData(verbose);
}
