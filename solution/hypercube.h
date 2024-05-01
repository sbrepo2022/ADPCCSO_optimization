#pragma once

#include <Eigen/Dense>


class Hypercube
{
public:
    Eigen::VectorXd base_point;
    double len;

    Hypercube(const Eigen::VectorXd &base_point, double len)
        : base_point(base_point), len(len)
    {}
};