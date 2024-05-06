#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <string>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <future>
#include <mutex>
#include <chrono>

#include "cxxopts/cxxopts.h"

#include "fitness_functions/alpine_1.h"
#include "fitness_functions/alpine_2.h"
#include "fitness_functions/deflected_corrugated_spring.h"
#include "fitness_functions/rastrigin.h"
#include "fitness_functions/spherical.h"

#include "agents_initializer.h"
#include "chicken_swarm.h"
#include "fish_swarm.h"
#include "helpers.h"
#include "agent.h"
#include "rootster.h"
#include "hen.h"
#include "chick.h"
#include "fish.h"


struct RunResult
{
    RunResult(
        Hypercube &limits,
        const std::vector<Eigen::VectorXd> &all_X,
        const std::vector<double> &all_fitness_vals,
        const std::vector<AgentClass> &agent_classes,
        const Eigen::VectorXd &best_X,
        double best_fitness_val,
        const std::vector<double> &stagnation
    )
        : limits(limits)
        , all_X(all_X)
        , all_fitness_vals(all_fitness_vals)
        , agent_classes(agent_classes)
        , best_X(best_X)
        , best_fitness_val(best_fitness_val)
        , stagnation(stagnation)
        {}

    Hypercube limits;

    std::vector<Eigen::VectorXd> all_X;
    std::vector<double> all_fitness_vals;
    std::vector<AgentClass> agent_classes;

    Eigen::VectorXd best_X;
    double best_fitness_val;

    std::vector<double> stagnation;
};


enum class GraphMode
{
    DRAW_2D,
    DRAW_3D
};


enum class SwarmUsage
{
    NONE=0x0,
    CHICKEN=0x1,
    FISH=0x2,
    BOTH=0x3
};


std::map<std::string, std::shared_ptr<FitnessFunction>> initFitnessFunctions();
cxxopts::ParseResult parseOptions(int argc, char *argv[], const std::map<std::string, std::shared_ptr<FitnessFunction>> &fitness_functions);
void printHelp(const cxxopts::Options &options);
SwarmUsage getSwarmUsage(const std::string &option);

RunResult run(size_t ndim, const std::shared_ptr<FitnessFunction> &fitness_function, SwarmUsage swarm_usage, double b_coef_opt, double d_coef_mult);
void swarmsExchange(
    const std::shared_ptr<ChickenSwarm> &chicken_swarm,
    const std::shared_ptr<FishSwarm> &fish_swarm,
    size_t num_agents_for_exchange
);

void saveResultX(const std::vector<RunResult> &run_results, bool best);
void printRunStatistics(const std::vector<RunResult> &run_results);
void draw2DGraphic(const std::vector<RunResult> &run_results, const std::shared_ptr<FitnessFunction> &fitness_function, bool best);
void drawStagnation(const std::vector<RunResult> &run_results);


std::map<std::string, std::shared_ptr<FitnessFunction>> initFitnessFunctions(size_t ndim)
{
    return {
        std::pair<std::string, std::shared_ptr<FitnessFunction>>("alpine_1", std::make_shared<fitness_function::Alpine1>(ndim)),
        std::pair<std::string, std::shared_ptr<FitnessFunction>>("alpine_2", std::make_shared<fitness_function::Alpine2>(ndim)),
        std::pair<std::string, std::shared_ptr<FitnessFunction>>("deflected_corrugated_spring", std::make_shared<fitness_function::DeflectedCorrugatedSpring>(ndim)),
        std::pair<std::string, std::shared_ptr<FitnessFunction>>("rastrigin", std::make_shared<fitness_function::Rastrigin>(ndim)),
        std::pair<std::string, std::shared_ptr<FitnessFunction>>("spherical", std::make_shared<fitness_function::Spherical>(ndim))
    };
}


cxxopts::ParseResult parseOptions(cxxopts::Options &options, int argc, char *argv[], const std::map<std::string, std::shared_ptr<FitnessFunction>> &fitness_functions)
{
    std::string fitness_help = "\t\tAvailable fitness functions: ";
    for (const auto& key : fitness_functions)
    {
        fitness_help += key.first + ", ";
    }

    std::string swarms_help = "\t\tAvailable swarms usage: chicken, fish, both";

    options.add_options()
        ("f,fitness", "Fitness function", cxxopts::value<std::string>()->default_value("spherical"), fitness_help)
        ("t,multistart-threads", "Number of parallel threads", cxxopts::value<int>()->default_value("1"))
        ("n,number-starts", "Number of starts", cxxopts::value<int>()->default_value("1"))
        ("b,best", "Write only best agent of each multistart", cxxopts::value<bool>()->default_value("true"))
        ("s,swarms", "Swarm usage", cxxopts::value<std::string>()->default_value("chicken"))
        ("b-coef", "b-coef", cxxopts::value<double>()->default_value("1.0"))
        ("d-coef-mult", "d-coef multiplier", cxxopts::value<double>()->default_value("1.0"))
        ("h,help", "Print usage")
        ;

    return options.parse(argc, argv);
}


void printHelp(const cxxopts::Options &options)
{
    std::cout << options.help() << std::endl;
    exit(0);
}


SwarmUsage getSwarmUsage(const std::string &option)
{
    if (option == "chicken") return SwarmUsage::CHICKEN;
    if (option == "fish") return SwarmUsage::FISH;
    if (option == "both") return SwarmUsage::BOTH;
    return SwarmUsage::NONE;
}


int main(int argc, char *argv[]) {
    srand(time(nullptr));

    // Подготовка
    size_t ndim = 2;
    std::map<std::string, std::shared_ptr<FitnessFunction>> fitness_functions = initFitnessFunctions(ndim);

    cxxopts::Options options("ADPCCSO_optimization", "Improved chicken swarm optimization method");
    cxxopts::ParseResult opt_parse_res;
    try
    {
        opt_parse_res = parseOptions(options, argc, argv, fitness_functions);
        if (opt_parse_res.count("help"))
        {
            printHelp(options);
        }
    }
    catch (const cxxopts::exceptions::no_such_option &e)
    {
        printHelp(options);
    }

    auto fitness_function = fitness_functions[opt_parse_res["fitness"].as<std::string>()];
    auto swarm_usage = getSwarmUsage(opt_parse_res["swarms"].as<std::string>());


    // Рассчеты
    std::vector<RunResult> run_results;
    std::mutex run_mtx;

    auto thread_task = [&]()
    {
        try
        {
            auto cur_res = run(ndim, fitness_function, swarm_usage, opt_parse_res["b-coef"].as<double>(), opt_parse_res["d-coef-mult"].as<double>());

            // Запись результатов
            std::lock_guard<std::mutex> lock(run_mtx);
            run_results.push_back(cur_res);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    };


    const int num_tasks = opt_parse_res["number-starts"].as<int>();
    const int num_threads = opt_parse_res["multistart-threads"].as<int>();

    std::vector<std::future<void>> futures;
    int task_count = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Пул потоков
    while (task_count < num_tasks) {
        int current_threads = 0; // Счетчик запущенных потоков

        // Запускаем задачи до достижения лимита одновременных потоков
        while (current_threads < num_threads && task_count < num_tasks) {
            futures.push_back(std::async(std::launch::async, thread_task));
            task_count++;
            current_threads++;
        }

        // Ожидаем завершение всех запущенных задач
        for (auto &fut : futures) {
            fut.get();
        }

        // Очищаем вектор futures для следующего цикла запуска задач
        futures.clear();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << std::endl << "Total execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;

    bool best = opt_parse_res["best"].as<bool>();
    saveResultX(run_results, best);
    printRunStatistics(run_results);
    if (ndim == 2) draw2DGraphic(run_results, fitness_function, best);

    return 0;
}


RunResult run(size_t ndim, const std::shared_ptr<FitnessFunction> &fitness_function, SwarmUsage swarm_usage, double b_coef_opt, double d_coef_mult)
{
    // Общие параметры
    size_t num_agents = 50;
    size_t num_agents_for_exchange = 1;
    size_t num_chicken_agents = swarm_usage == SwarmUsage::BOTH ? 25 : (swarm_usage == SwarmUsage::CHICKEN ? num_agents : 0);
    size_t num_fish_agents = swarm_usage == SwarmUsage::BOTH ? 25 : (swarm_usage == SwarmUsage::FISH ? num_agents : 0);
    size_t max_gen = 1000;

    // Параметры куриного роя
    double rootsters_coef = 0.2;
    double hens_coef = 0.6;
    double learn_factor_min = 0.4;
    double learn_factor_max = 0.9;

    // Параметры роя рыб
    double fish_step = 0.3;
    double fish_visual = 2.5;

    auto hypercube = fitness_function->getBoundHypercube();

    double b_coef = b_coef_opt;
    double d_coef = math_helpers::calcDCoef(hypercube.len, num_agents, ndim) * d_coef_mult;

    fitness_function->setBCoef(b_coef);
    fitness_function->setDCoef(d_coef);

    // Инициализация куриного роя
    std::shared_ptr<ChickenSwarm> chicken_swarm;
    if ((size_t)swarm_usage & (size_t)SwarmUsage::CHICKEN)
    {
        std::vector<Eigen::VectorXd> X_chicken = AgentInitializer::hypercubeUniformInitializer(hypercube, num_chicken_agents);
        chicken_swarm = std::make_shared<ChickenSwarm>(
            fitness_function,
            rootsters_coef * num_chicken_agents,
            hens_coef * num_chicken_agents,
            num_chicken_agents - rootsters_coef * num_chicken_agents - hens_coef * num_chicken_agents,
            max_gen,
            learn_factor_min,
            learn_factor_max
        );
        chicken_swarm->startupAgentsInit(X_chicken);
    }

    // Инициализация роя рыб
    std::shared_ptr<FishSwarm> fish_swarm;
    if ((size_t)swarm_usage & (size_t)SwarmUsage::FISH)
    {
        std::vector<Eigen::VectorXd> X_fish = AgentInitializer::hypercubeUniformInitializer(hypercube, num_fish_agents);
        fish_swarm = std::make_shared<FishSwarm>(fitness_function, fish_step, fish_visual);
        fish_swarm->startupAgentsInit(X_fish);
    }

    // Алгоритм локализации минимума
    auto calc_sync_gen = [](size_t t) -> size_t
    {
        return std::round(40 + 60 * (1 + exp(15 - 0.5*t)));
    };

    size_t sync_gen = calc_sync_gen(0);

    std::vector<double> stagnation;
    for (size_t cur_gen = 1; cur_gen < max_gen; cur_gen++)
    {
        sync_gen = calc_sync_gen(cur_gen);

        double chicken_best = std::numeric_limits<double>::max();
        double fish_best = std::numeric_limits<double>::max();

        if ((size_t)swarm_usage & (size_t)SwarmUsage::CHICKEN)
            chicken_swarm->doMove([&chicken_swarm, cur_gen, sync_gen](size_t i) {
                return chicken_swarm->getAgents()[i]->calcMove(cur_gen, cur_gen % sync_gen == 0);
            });
            chicken_best = chicken_swarm->getOptimalValue();

        if ((size_t)swarm_usage & (size_t)SwarmUsage::FISH)
            fish_swarm->doMove([&fish_swarm, cur_gen, sync_gen](size_t i) {
                return fish_swarm->getAgents()[i]->calcMove(cur_gen, cur_gen % sync_gen == 0);
            });
            fish_best = fish_swarm->getOptimalValue();

        if (swarm_usage == SwarmUsage::BOTH)
            swarmsExchange(chicken_swarm, fish_swarm, num_agents_for_exchange);
            chicken_best = chicken_swarm->getOptimalValue();
            fish_best = fish_swarm->getOptimalValue();

        if ((size_t)swarm_usage & (size_t)SwarmUsage::CHICKEN)
            if (cur_gen % sync_gen == 0)
            {
                chicken_swarm->updateAgentsRoles();
            }

        stagnation.push_back(std::min(chicken_best, fish_best));
    }

    constexpr bool print_verbose = true;
    if ((size_t)swarm_usage & (size_t)SwarmUsage::CHICKEN) chicken_swarm->printData(print_verbose);
    if ((size_t)swarm_usage & (size_t)SwarmUsage::FISH) fish_swarm->printData(print_verbose);

    std::vector<Eigen::VectorXd> all_X;
    std::vector<double> all_fitness_vals;
    std::vector<AgentClass> agent_classes;

    if ((size_t)swarm_usage & (size_t)SwarmUsage::CHICKEN)
    {
        for (auto& agent : chicken_swarm->getAgents())
        {
            all_X.push_back(agent->getX());
            all_fitness_vals.push_back(agent->getCachedFitnessValue());
            agent_classes.push_back(agent->getAgentClass());
        }
    }

    if ((size_t)swarm_usage & (size_t)SwarmUsage::FISH)
    {
        for (auto& agent : fish_swarm->getAgents())
        {
            all_X.push_back(agent->getX());
            all_fitness_vals.push_back(agent->getCachedFitnessValue());
            agent_classes.push_back(agent->getAgentClass());
        }
    }

    if (swarm_usage == SwarmUsage::BOTH)
    {
        if (chicken_swarm->getOptimalValue() < fish_swarm->getOptimalValue())
            return RunResult(hypercube, all_X, all_fitness_vals, agent_classes, chicken_swarm->getOptimalX(), chicken_swarm->getOptimalValue(), stagnation);
        else
            return RunResult(hypercube, all_X, all_fitness_vals, agent_classes, fish_swarm->getOptimalX(), fish_swarm->getOptimalValue(), stagnation);
    }
    else if (swarm_usage == SwarmUsage::CHICKEN) return RunResult(hypercube, all_X, all_fitness_vals, agent_classes, chicken_swarm->getOptimalX(), chicken_swarm->getOptimalValue(), stagnation);
    else if (swarm_usage == SwarmUsage::FISH) return RunResult(hypercube, all_X, all_fitness_vals, agent_classes, fish_swarm->getOptimalX(), fish_swarm->getOptimalValue(), stagnation);
    else return RunResult(hypercube, {}, {}, {}, Eigen::VectorXd::Zero(ndim), 0.0, {});
}


void swarmsExchange(
    const std::shared_ptr<ChickenSwarm> &chicken_swarm,
    const std::shared_ptr<FishSwarm> &fish_swarm,
    size_t num_agents_for_exchange
)
{
    // Обмен значениями
    size_t chicken_best_agent_i = chicken_swarm->getOptimalAgentIndex();
    size_t fish_best_agent_i = fish_swarm->getOptimalAgentIndex();

    std::deque<size_t> chicken_exchange_indices(chicken_swarm->getAgents().size());
    std::deque<size_t> fish_exchange_indices(fish_swarm->getAgents().size());
    std::iota(chicken_exchange_indices.begin(), chicken_exchange_indices.end(), 0);
    std::iota(fish_exchange_indices.begin(), fish_exchange_indices.end(), 0);

    chicken_exchange_indices.erase(chicken_exchange_indices.begin() + chicken_best_agent_i);
    fish_exchange_indices.erase(fish_exchange_indices.begin() + fish_best_agent_i);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(chicken_exchange_indices.begin(), chicken_exchange_indices.end(), g);
    std::shuffle(fish_exchange_indices.begin(), fish_exchange_indices.end(), g);

    chicken_exchange_indices.resize(num_agents_for_exchange - 1);
    fish_exchange_indices.resize(num_agents_for_exchange - 1);

    chicken_exchange_indices.push_front(chicken_best_agent_i);
    fish_exchange_indices.push_front(fish_best_agent_i);

    // Debug
    // constexpr bool print_verbose = true;
    // chicken_swarm->printData(print_verbose);
    // fish_swarm->printData(print_verbose);

    // std::cout << "Chickens for exchange: [ ";
    // for (auto i : chicken_exchange_indices) std::cout << i << " ";
    // std::cout << "]\n";

    // std::cout << "Fish for exchange: [ ";
    // for (auto i : fish_exchange_indices) std::cout << i << " ";
    // std::cout << "]\n";
    // End debug

    for (size_t i = 0; i < num_agents_for_exchange; i++)
    {
        auto swap_X = chicken_swarm->getAgents()[chicken_exchange_indices[i]]->getX();
        chicken_swarm->getAgents()[chicken_exchange_indices[i]]->updateX(
            fish_swarm->getAgents()[fish_exchange_indices[i]]->getX()
        );
        fish_swarm->getAgents()[fish_exchange_indices[i]]->updateX(swap_X);
    }

    chicken_swarm->doMove([&](size_t i) {
        return chicken_swarm->getAgents()[i]->getX();
    });
    fish_swarm->doMove([&](size_t i) {
        return fish_swarm->getAgents()[i]->getX();
    });
}


void saveResultX(const std::vector<RunResult> &run_results, bool best)
{
    std::ofstream ofile;
    ofile.open("results.csv");

    for (auto& run_res : run_results)
    {
        if (best)
        {
            for (int j = 0; j < run_res.best_X.size(); j++)
            {
                ofile << run_res.best_X[j];

                if (j < run_res.best_X.size() - 1)
                {
                    ofile << ",";
                }
                else
                {
                    ofile << "\n";
                }
            }
        }
        else
        {
            for (int i = 0; i < run_res.all_X.size(); i++)
            {
                for (int j = 0; j < run_res.all_X[i].size(); j++)
                {
                    if (! run_res.limits.isXIn(run_res.all_X[i])) continue;

                    ofile << run_res.all_X[i][j];

                    if (j < run_res.all_X[i].size() - 1)
                    {
                        ofile << ",";
                    }
                    else
                    {
                        ofile << "\n";
                    }
                }
            }
        }
    }
}


void printRunStatistics(const std::vector<RunResult> &run_results)
{
    std::cout << "\nRun statistics:\n";
    double sum = 0;

    std::cout << "Points groups (acceptable - 0, almost_acceptable - 1, unacceptable - 2):\n[ ";
    for (auto& run_res : run_results)
    {
        for (auto& ac : run_res.agent_classes)
        {
            std::cout << (int)ac << " ";
        }
    }
    std::cout << "]\n";
}


void draw2DGraphic(const std::vector<RunResult> &run_results, const std::shared_ptr<FitnessFunction> &fitness_function, bool best)
{
    // Сохранение точек в файл
    std::ofstream pointsFile("points.dat");
    for (const auto& run_res : run_results) {
        if (best)
        {
            pointsFile << run_res.best_X[0] << " " << run_res.best_X[1] << " " << run_res.best_fitness_val << "\n";
        }
        else
        {
            for (int i = 0; i < run_res.all_X.size(); i++)
            {
                if (run_res.limits.isXIn(run_res.all_X[i]))
                {
                    pointsFile << run_res.all_X[i][0] << " " << run_res.all_X[i][1] << " " << run_res.all_fitness_vals[i] << "\n";
                }
            }
        }
    }
    pointsFile.close();

    // Сохранение графика фитнесс-функции
    std::ofstream data_file("fitness.dat");
    int nx = 100;
    int ny = 100;
    double side_len = fitness_function->getBoundHypercube().len;
    double base_x = fitness_function->getBoundHypercube().base_point[0];
    double base_y = fitness_function->getBoundHypercube().base_point[1];
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            double x = base_x + (j/(double)(nx-1)) * side_len;
            double y = base_y + (i/(double)(ny-1)) * side_len;

            Eigen::VectorXd X(2);
            X << x, y;

            double value = fitness_function->fitness(X);
            data_file << x << " " << y << " " << value << "\n";
        }
        data_file << "\n";
    }
    data_file.close();

    // Создание скрипта Gnuplot
    auto createGraphFile = [](GraphMode graph_mode)
    {
        std::ofstream script_file;
        if (graph_mode == GraphMode::DRAW_2D)
            script_file.open("plot_script_2d.gp");
        else
            script_file.open("plot_script_3d.gp");

        if (graph_mode == GraphMode::DRAW_2D)
            script_file << "set pm3d map\n";
        else
            script_file << "set pm3d\n";

        script_file << "set palette rgbformulae 33,13,10\n";
        script_file << "set xlabel 'X[0]'\n";
        script_file << "set ylabel 'X[1]'\n";
        script_file << "set title 'Heatmap of the Fitness Function'\n";

        if (graph_mode == GraphMode::DRAW_2D)
            script_file << "splot 'fitness.dat' using 1:2:3 with image, 'points.dat' using 1:2:3 with points pt 7 ps 1.5 lc 'red' title 'Points'\n";
        else
            script_file << "splot 'fitness.dat' using 1:2:3 with pm3d, 'points.dat' using 1:2:3 with points pt 7 ps 1.5 lc 'red' title 'Points'\n";

        script_file << "pause -1\n";
        script_file.close();
    };

    createGraphFile(GraphMode::DRAW_2D);
    createGraphFile(GraphMode::DRAW_3D);
}


void drawStagnationGraphic(const std::vector<RunResult> &run_results)
{
    std::vector<std::vector<double>> stagnation;

    bool continue_filling = true;
    int gen_i = 1;
    while (continue_filling)
    {
        std::vector<double> stagnation_row;
        for (const auto& run_res : run_results) {
            if (run_res.stagnation.size() - 1 == gen_i) continue_filling = false;

            stagnation_row.push_back(run_res.stagnation[gen_i]);
        }
        stagnation.push_back(stagnation_row);
        gen_i++;
    }

    std::ofstream data_file("stagnation.dat");
    for (int i = 0; i < stagnation.size(); i++)
    {
        data_file << i << " ";
        for (int j = 0; j < stagnation[i].size(); j++)
        {
            data_file << stagnation[i][j] << " ";
        }
        data_file << std::endl;
    }
    data_file.close();

    std::ofstream script_file("plot_script_stagnation.gp");
    script_file << "plot ";
    for (int i = 0; i < run_results.size(); i++)
    {
        script_file << "stagnation.dat using 1:" << i + 2 << " with lines, \\\n";
    }
    script_file << "pause -1\n";
    script_file.close();
}
