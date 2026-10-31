#include "functionClasses.h"
#include <cmath>

using namespace fct;

double Constant::operator() (double temp, const std::vector<double>& func_params){return func_params[0];}

Briere::Briere(double scaling = 1.0):scaling(scaling){}
double Briere::operator() (double temp, const std::vector<double>& func_params){
    // func_params = q, Tmin, Tmax
    if(temp>std::max(0.0, func_params[1]) && temp<func_params[2]){
        return func_params[0]/scaling*temp*(temp-func_params[1])*std::sqrt(func_params[2]-temp);
    }
    else{
        return 0;
    }
}

Quadratic::Quadratic(double scaling = 1.0):scaling(scaling){}
double Quadratic::operator() (double temp, const std::vector<double>& func_params){
    // q, Tmin, Tmax = func_parameters
    if(temp>func_params[1] && temp<func_params[2]){
        double quadratic_expression = func_params[0]/scaling*(temp-func_params[1])*(func_params[2]-temp);
        return std::min(quadratic_expression, 1.0);
    }
    else{return 0;}
}

Pol4::Pol4(double scaling = 1.0):scaling(scaling){}
double Pol4::operator() (double temp, const std::vector<double>& func_params){
    //q, Tmin, Tmax, offset = func_params
    // temp = std::max(func_params[1],temp);
    // temp = std::min(func_params[2],temp);
    // if(temp>=func_params[1] && temp<=func_params[2]){
    double expression = func_params[0]/scaling*std::pow((temp-(func_params[1]+func_params[2])/2),4) + func_params.at(3);
    double posExpression = std::max(expression, 0.0);
    // return std::min(posExpression, 10.0);
    return posExpression;
    // }
    // else{return 10.0;}
}

TruncatedLinear::TruncatedLinear(double scaling = 1.0):scaling(scaling){}
double TruncatedLinear::operator() (double temp, const std::vector<double>& func_params){
    // beta, Tmax = func_params
    if(temp<func_params[1]){
        // plateau at 14 degrees, this is the lowest temperature in the datasets used by global health paper. they also do this
        temp = std::max(temp, 14.0);
        return -func_params[0]/scaling*(temp-func_params[1]);
    }
    else{return 0;}
}

double Sigmoidal::operator() (double temp, const std::vector<double>& func_params){
    // func_params = alpha, beta
    return 1/(1+std::exp(-(func_params[1]*temp + func_params[0])));
}

double ExponentialDecay::operator() (double temp, const std::vector<double>& func_params){
    // alpha, beta = func_params
    return std::exp(-func_params[1]*temp + func_params[0]); 
}