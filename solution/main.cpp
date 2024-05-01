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
#include "helpers.h"
#include "agent.h"
#include "rootster.h"
#include "hen.h"
#include "chick.h"


struct RunResult
{
    RunResult(Eigen::VectorXd best_X, double best_fitness_val)
        : best_X(best_X), best_fitness_val(best_fitness_val)
        {}

    Eigen::VectorXd best_X;
    double best_fitness_val;
};


enum class GraphMode
{
    DRAW_2D,
    DRAW_3D
};


std::map<std::string, std::shared_ptr<FitnessFunction>> initFitnessFunctions();
cxxopts::ParseResult parseOptions(int argc, char *argv[], const std::map<std::string, std::shared_ptr<FitnessFunction>> &fitness_functions);
void printHelp(const cxxopts::Options &options);
RunResult run(size_t ndim, const std::shared_ptr<FitnessFunction> &fitness_function);
void saveResultX(const std::vector<RunResult> &run_results);
void printRunStatistics(const std::vector<RunResult> &run_results);
void draw2DGraphic(const std::vector<RunResult> &run_results, const std::shared_ptr<FitnessFunction> &fitness_function);


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

    options.add_options()
        ("f,fitness", "Fitness function", cxxopts::value<std::string>()->default_value("spherical"), fitness_help)
        ("h,help", "Print usage")
        ;

    return options.parse(argc, argv);
}


void printHelp(const cxxopts::Options &options)
{
    std::cout << options.help() << std::endl;
    exit(0);
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


    // Рассчеты
    std::vector<RunResult> run_results;
    std::mutex run_mtx;

    auto thread_task = [&]()
    {
        try
        {
            auto cur_res = run(ndim, fitness_function);

            // Запись результатов
            std::lock_guard<std::mutex> lock(run_mtx);
            run_results.push_back(cur_res);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    };


    const int num_tasks = 32;
    const int num_threads = 8;

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

    saveResultX(run_results);
    printRunStatistics(run_results);
    if (ndim == 2) draw2DGraphic(run_results, fitness_function);

    return 0;
}


RunResult run(size_t ndim, const std::shared_ptr<FitnessFunction> &fitness_function)
{
    size_t num_agents = 200;
    double rootsters_coef = 0.2;
    double hens_coef = 0.6;
    size_t max_gen = 1000;
    double learn_factor_min = 0.4;
    double learn_factor_max = 0.9;

    auto hypercube = fitness_function->getBoundHypercube();

    double b_coef = 1;
    double d_coef = math_helpers::calcDCoef(hypercube.len, num_agents, ndim);

    fitness_function->setBCoef(b_coef);
    fitness_function->setDCoef(d_coef);

    std::vector<Eigen::VectorXd> X = AgentInitializer::hypercubeUniformInitializer(hypercube, num_agents);
    auto chicken_swarm = std::make_shared<ChickenSwarm>(
        fitness_function,
        rootsters_coef * num_agents,
        hens_coef * num_agents,
        num_agents - rootsters_coef * num_agents - hens_coef * num_agents,
        max_gen,
        learn_factor_min,
        learn_factor_max
    );
    chicken_swarm->startupAgentsInit(X);

    // Проверка инициализации
    // for (auto& agent : chicken_swarm->getAgents())
    // {
    //     std::cout << "Agent [" << agent->getAgentIndex() <<  "] has type <" << agent->getAgentType() << ">" << std::endl;
    // }

    // Алгоритм локализации минимума
    auto calc_sync_gen = [](size_t t) -> size_t
    {
        return std::round(40 + 60 * (1 + exp(15 - 0.5*t)));
    };

    size_t sync_gen = calc_sync_gen(0);

    for (size_t cur_gen = 1; cur_gen < max_gen; cur_gen++)
    {
        sync_gen = calc_sync_gen(cur_gen);
        chicken_swarm->doMove(cur_gen, cur_gen % sync_gen == 0);
        if (cur_gen % sync_gen == 0)
        {
            chicken_swarm->updateAgentsRoles();
        }
    }

    constexpr bool print_verbose = false;
    chicken_swarm->printData(print_verbose);

    return RunResult(chicken_swarm->getOptimalX(), chicken_swarm->getOptimalValue());
}


void saveResultX(const std::vector<RunResult> &run_results)
{
    std::ofstream ofile;
    ofile.open("results.csv");

    for (auto& run_res : run_results)
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
}


void printRunStatistics(const std::vector<RunResult> &run_results)
{
    double sum = 0;

    for (auto& run_res : run_results)
    {

    }
}


void draw2DGraphic(const std::vector<RunResult> &run_results, const std::shared_ptr<FitnessFunction> &fitness_function)
{
    // Сохранение точек в файл
    std::ofstream pointsFile("points.dat");
    for (const auto& run_res : run_results) {
        pointsFile << run_res.best_X[0] << " " << run_res.best_X[1] << " " << run_res.best_fitness_val << "\n";
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
