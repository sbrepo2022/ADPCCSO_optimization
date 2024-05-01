#pragma once

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

namespace math_helpers
{

double randn(double sigma);
double uniform(double min = 0.0, double max = 1.0);
std::vector<size_t> randomParentChildAssignments(double num_childs, double num_parents);
double calcDCoef(double cube_edge_len, size_t num_agents, size_t ndim);

}
