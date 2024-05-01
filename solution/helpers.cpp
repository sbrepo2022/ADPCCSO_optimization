#include "helpers.h"

namespace math_helpers
{

double randn(double sigma)
{
    const gsl_rng_type* T;
    gsl_rng* r;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    double value = gsl_ran_gaussian(r, sigma);

    gsl_rng_free(r);

    return value;
}


double uniform(double min, double max)
{
  return min + rand() / (double)RAND_MAX * (max - min);
}


std::vector<size_t> randomParentChildAssignments(double num_childs, double num_parents)
{
    size_t N = num_childs; // количество детей
    size_t M = num_parents;  // количество родителей

    // Вычисляем базовое число детей на родителя и сколько родителей получат на одного ребенка больше
    size_t base_children = N / M;
    size_t extra_children_count = N % M;

    // Создаем вектор, в котором каждый элемент показывает сколько детей будет у каждого родителя
    std::vector<size_t> parents_children_count(M, base_children);
    for (size_t i = 0; i < extra_children_count; ++i) {
        parents_children_count[i]++;
    }

    // Перемешиваем счетчики числа детей у родителей
    std::random_device rd1;
    std::mt19937 g1(rd1());
    std::shuffle(parents_children_count.begin(), parents_children_count.end(), g1);

    // Создаем вектор, где каждый родитель записан столько раз, сколько у него детей
    std::vector<size_t> assignments;
    for (size_t parent = 0; parent < M; ++parent) {
        assignments.insert(assignments.end(), parents_children_count[parent], parent);
    }

    // Перемешиваем родителей
    std::random_device rd2;
    std::mt19937 g2(rd2());
    std::shuffle(assignments.begin(), assignments.end(), g2);

    return assignments;
}


double calcDCoef(double cube_edge_len, size_t num_agents, size_t ndim)
{
    return cube_edge_len / pow(num_agents, 1/ndim) * sqrt(ndim);
}

}