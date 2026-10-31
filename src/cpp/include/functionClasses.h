#pragma once
#include<vector>

namespace fct{


struct Constant{
    double operator()(double temp, const std::vector<double>& func_params);
};

struct Briere{
    double scaling{1.0};
    Briere(double);
    double operator() (double temp, const std::vector<double>& func_params);
};

struct Quadratic{
    double scaling{1.0};
    Quadratic(double);
    double operator() (double temp, const std::vector<double>& func_params);
};

struct Pol4{
    double scaling{1.0};
    Pol4(double);
    double operator() (double temp, const std::vector<double>& func_params);
};

struct TruncatedLinear{
    double scaling{1.0};
    TruncatedLinear(double);
    double operator() (double temp, const std::vector<double>& func_params);
};

struct Sigmoidal{
    double operator() (double temp, const std::vector<double>& func_params);
};

struct ExponentialDecay{
    double operator() (double temp, const std::vector<double>& func_params);
};


}