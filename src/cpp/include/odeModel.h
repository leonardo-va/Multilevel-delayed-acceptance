#pragma once

// #include <boost/filesystem.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/function.hpp>
#include <boost/random.hpp>
#include <fstream>
#include <map>
#include <set>
#include <types.h>

typedef std::vector< double > state_type;
typedef boost::function<double (double temp, const std::vector<double>& func_params)> seir_param;

class EvaluatorBase;

struct solverSettings{
    solverSettings(double start, double end, double stepsize, state_type initialState);
    double startTime = 0.0;
    double endTime = 1.0;
    double stepSize = 0.001;
    state_type initialState;
    int dim;
    int nSteps;
    std::vector<double> solverGrid;
};

class SEIRModel {
    
    public:
    
    enum class ParamName{delta_E, p_E, mu_E, alpha, delta_J, p_J, mu_J, omega, 
                          lf_M, a, beta_er, b_M, EIP_M, Lambda, b_H, mu_H, alpha_H, gamma_H, count=18};
    
                          
    static const inline std::map<ParamName, size_t> ParamSizes{
        {ParamName::delta_E, 3},
        {ParamName::p_E, 3},
        {ParamName::mu_E, 4},
        {ParamName::alpha, 1},
        {ParamName::delta_J, 3},
        {ParamName::p_J, 3},
        {ParamName::mu_J, 4},
        {ParamName::omega, 1},
        {ParamName::lf_M, 2},
        {ParamName::a, 3},
        {ParamName::beta_er, 1},
        {ParamName::b_M, 2},
        {ParamName::EIP_M, 2},
        {ParamName::Lambda, 1},
        {ParamName::b_H, 1},
        {ParamName::mu_H, 1},
        {ParamName::alpha_H, 1},
        {ParamName::gamma_H, 1}
    };
    
    static const inline std::map<ParamName, std::string> ParamStrings{
        {ParamName::delta_E, "delta_E"},
        {ParamName::p_E, "p_E"},
        {ParamName::mu_E, "mu_E"},
        {ParamName::alpha, "alpha"},
        {ParamName::delta_J, "delta_J"},
        {ParamName::p_J, "p_J"},
        {ParamName::mu_J, "mu_J"},
        {ParamName::omega, "omega"},
        {ParamName::lf_M, "lf_M"},
        {ParamName::a, "a"},
        {ParamName::beta_er, "beta_er"},
        {ParamName::b_M, "b_M"},
        {ParamName::EIP_M, "EIP_M"},
        {ParamName::Lambda, "Lambda"},
        {ParamName::b_H, "b_H"},
        {ParamName::mu_H, "mu_H"},
        {ParamName::alpha_H, "alpha_H"},
        {ParamName::gamma_H, "gamma_H"}
    };
    
    static const inline std::vector<ParamName> ModelParameters{
        ParamName::delta_E, ParamName::p_E, ParamName::mu_E,ParamName::alpha, ParamName::delta_J, ParamName::p_J, ParamName::mu_J,ParamName::omega, 
        ParamName::lf_M, ParamName::a, ParamName::beta_er, ParamName::b_M, ParamName::EIP_M, ParamName::Lambda, 
        ParamName::b_H, ParamName::mu_H, ParamName::alpha_H, ParamName::gamma_H
    };
    
    std::map<ParamName, seir_param> ModelParameterFunctions{
        {ParamName::delta_E, seir_param{}},
        {ParamName::p_E, seir_param{}},
        {ParamName::mu_E, seir_param{}},
        {ParamName::alpha, seir_param{}},
        {ParamName::delta_J, seir_param{}},
        {ParamName::p_J, seir_param{}},
        {ParamName::mu_J, seir_param{}},
        {ParamName::omega, seir_param{}},
        {ParamName::lf_M, seir_param{}},
        {ParamName::a, seir_param{}},
        {ParamName::beta_er, seir_param{}},
        {ParamName::b_M, seir_param{}},
        {ParamName::EIP_M, seir_param{}},
        {ParamName::Lambda, seir_param{}},
        {ParamName::b_H, seir_param{}},
        {ParamName::mu_H, seir_param{}},
        {ParamName::alpha_H, seir_param{}},
        {ParamName::gamma_H, seir_param{}}
    };
    
    SEIRModel();
    // SEIRModel(const SEIRModel&);
    /// @brief clones the model, without cached parameters.
    /// @return 
    SEIRModel clone() const;
    double getCachedTime(){ return m_cachedTime;};
    void eval(state_type &x, state_type &dxdt, const double t );
    
    void updateCachedParameters(double t, const std::vector<ParamName>& whichOnes);
    
    std::vector<double>& get_function_param(ParamName name);
    
    std::vector<double> get_function_param(ParamName name) const;

    const std::map<ParamName, seir_param> getAllFunctions() const;
    const std::vector<std::vector<double>> getAllFunctionParameters() const;
    const boost::function<double(double t)> get_temperature_profile() const { return temperature_profile;};
    
    void set_function_parameter(ParamName name, const std::vector<double> &value);

    void set_cacheUpdateTolerance(double time){m_cacheUpdateTolerance = time;}

    void set_evaluator(std::shared_ptr<EvaluatorBase> evaluator){m_evaluator = evaluator;}
    
    private:
    double m_cachedTime;
    double m_cachedTemperature;
    std::map<ParamName, double> m_cachedParameters;
    std::vector<double> m_indexedCache;
    double m_cacheUpdateTolerance;
    std::shared_ptr<EvaluatorBase> m_evaluator = nullptr;
    boost::function<double(double t)> temperature_profile;
    
    // these are groundtruth values from paper "......."
    std::vector<double> fp_delta_E{3.43, 2.53, 51.69};
    std::vector<double> fp_p_E{5.70, 6.62, 33.29};
    std::vector<double> fp_mu_E{5.76,6.98,26.66,0.21};
    std::vector<double> fp_alpha{2*std::pow(10,-2)};
    std::vector<double> fp_delta_J{3.96, 1.57, 41.9};
    std::vector<double> fp_p_J{4.41, 7.38, 36.14};
    std::vector<double> fp_mu_J{5.39,7.64,28.74,0.33};
    std::vector<double> fp_omega{0.5};
    std::vector<double> fp_lf_M{4.42, 34.05};
    std::vector<double> fp_a{3.71, 1.93, 45.1};
    std::vector<double> fp_beta_er{140};
    std::vector<double> fp_b_M{-4.64, 0.25};
    std::vector<double> fp_EIP_M{5.36, 0.08};
    std::vector<double> fp_Lambda{5};
    std::vector<double> fp_b_H{0.8};
    std::vector<double> fp_mu_H{0.001};
    std::vector<double> fp_alpha_H{0.5};
    std::vector<double> fp_gamma_H{0.25};
    
    // values from paper
    double scaling_delta_E{1.6*std::pow(10,4)};
    // double scaling_p_E{1000.};
    double scaling_p_E{1600.};
    double scaling_mu_E{100000.};
    double scaling_alpha{1.};
    double scaling_delta_J{1.6*std::pow(10,4)};
    // double scaling_p_J{1000.};
    double scaling_p_J{1600.};
    double scaling_mu_J{100000.};
    double scaling_omega{1.};
    double scaling_lf_M{150.};
    double scaling_a{43000.};
    double scaling_beta_er{1.};
    double scaling_b_M{1.};
    double scaling_EIP_M{1.};
    double scaling_Lambda{1.};
    double scaling_b_H{1.};
    double scaling_mu_H{1.};
    double scaling_alpha_H{1.};
    double scaling_gamma_H{1.};
    
};

template <typename T>
constexpr auto parameterIndex(T value) {
    return static_cast<std::size_t>(value);
}

class EvaluatorBase {
public:
    virtual ~EvaluatorBase() = default;

    virtual void evaluate(
        const state_type& x,
        state_type& dxdt,
        const double t,
        std::vector<double>& cache
    ) = 0;
    virtual const std::vector<SEIRModel::ParamName>& getTempDependentParameters() const = 0;
};

class FullModelEvaluator : public EvaluatorBase{
    public:
        void evaluate(const state_type&x, state_type& dxdt, const double t, std::vector<double>& cache) override;
        const std::vector<SEIRModel::ParamName>& getTempDependentParameters() const override {return m_tempDependentParameters;};
    private:
        std::vector<SEIRModel::ParamName> m_tempDependentParameters{
            SEIRModel::ParamName::delta_E, 
            SEIRModel::ParamName::p_E, 
            SEIRModel::ParamName::mu_E,
            SEIRModel::ParamName::delta_J,
            SEIRModel::ParamName::p_J,
            SEIRModel::ParamName::mu_J,
            SEIRModel::ParamName::lf_M,
            SEIRModel::ParamName::a,
            SEIRModel::ParamName::b_M,
            SEIRModel::ParamName::EIP_M
        };
};

// class MosquitoPopulationEvaluator : public EvaluatorBase{
//     public:
//         void evaluate(const state_type&x, state_type& dxdt, const double t, std::vector<double>& cache) override;
//         const std::vector<SEIRModel::ParamName>& getTempDependentParameters() const override {return m_tempDependentParameters;};
//     private:
//         std::vector<SEIRModel::ParamName> m_tempDependentParameters{
//             SEIRModel::ParamName::delta_E, 
//             SEIRModel::ParamName::mu_E,
//             SEIRModel::ParamName::delta_J,
//             SEIRModel::ParamName::mu_J,
//             SEIRModel::ParamName::lf_M,
//             SEIRModel::ParamName::a,
//         };
// };

class SEIRObserver
{   
    public:
    std::vector< state_type >& m_states;
    std::vector< double >& m_times;
    size_t m_current_index{0};
    std::vector<std::vector<double>> m_qoi_linear_combinations;
    
    
    // expect 0-initialized states
    SEIRObserver( std::vector< state_type > &states , std::vector< double > &times);
    
    void add_qoi(const std::vector<double> &lc_qoi);
    
    void observe_qois( const state_type &x, double t);
    
    void observe_full( const state_type &x, double t);
    
    void writeResult(std::string path);
    
    size_t number_of_quantities();
    
    void restart();
    
};

class ParameterDescription{
    
    public:
    ParameterDescription(SEIRModel::ParamName name, std::set<int> subsetIndices = std::set<int>{}):
        m_name(name), 
        m_name_str(SEIRModel::ParamStrings.at(name)),
        m_totalSize(SEIRModel::ParamSizes.at(name)), 
        m_subsetIndices(std::vector<int>{subsetIndices.begin(), subsetIndices.end()})
        { validate(); };
    size_t size() const;
    std::set<int> getSubsetIndices() const;
    const SEIRModel::ParamName m_name;
    const std::string m_name_str;
    const size_t m_totalSize;
    const std::vector<int> m_subsetIndices;
    bool isSubset(const ParameterDescription& supersetDesc) const;
    bool isSuperset(const ParameterDescription& subsetDesc) const;

    private:  
    void validate();
};

bool operator==(const ParameterDescription& desc1, const ParameterDescription& desc2);

std::vector<double> lc_qoi(const std::vector<double>& linearCombination, const std::vector<state_type>& solution);

std::vector<state_type> generate_data(const SEIRObserver& observer, std::vector<double> stddev_vector, bool cutoffZero = false);
std::vector<state_type> generate_data(const std::vector<state_type>& states, std::vector<double> stddev_vector, bool cutoffZero = false);
std::vector<state_type> generate_data_sparse(const std::vector<state_type>& states, std::vector<double> stddev_vector, double fractionOfZeros);

void apply_qois(const state_type& in_state, const std::vector<std::vector<double>>& linear_combinations, state_type& out_state);

std::vector<double> make_grid(double startTime, double endTime, size_t n_measurements);

/// @brief solve model with rk 4 method.
/// @param model 
/// @param measurementTimes points where to evaluate the solution
/// @param stepsize 
std::vector< state_type > odeSolve(SEIRModel& model, std::vector<double> measurementTimes, std::vector<double> initialValue, double stepsize, std::optional<std::string> writeCsv = std::optional<std::string>{});

void evaluateModelParameters(const SEIRModel& model, std::string outFolder);

std::vector<double> calculateR0(SEIRModel& model, solverSettings settings);