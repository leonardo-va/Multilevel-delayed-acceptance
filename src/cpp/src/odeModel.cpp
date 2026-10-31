#include <odeModel.h>
#include <functionClasses.h>
#include <random>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#include <util.h>
#include <filesystem>
#include <io.h>


std::vector<double> lc_qoi(const std::vector<double>& linearCombination, const std::vector<state_type>& solution){
    if(solution[0].size() != linearCombination.size()){
        throw std::runtime_error("Wrong size of linear combination");
    }
    std::vector<double> qoi(solution.size(), 0.0);
    for(int j = 0; j < solution.size(); ++j){

        for(int i=0; i<linearCombination.size(); ++i){
            qoi[j] += linearCombination[i]*solution[j][i];
        }
    }
    return qoi;
}

std::vector<state_type> generate_data(const SEIRObserver& observer, std::vector<double> stddev_vector, bool cutoffZero){
    return generate_data(observer.m_states, stddev_vector, cutoffZero);
}
std::vector<state_type> generate_data(const std::vector<state_type>& states, std::vector<double> stddev_vector, bool cutoffZero){
    
    // std::vector<double> qoi = lc_qoi(linearCombination, observer.m_states);
    size_t n_qois = states.at(0).size();
    std::vector<state_type> qoi_noisy = states;
    boost::random::mt19937 rng(std::random_device{}());
    std::vector<boost::random::normal_distribution<double>> noise_generators;
    for(int i = 0; i<n_qois; ++i){
        boost::random::normal_distribution<double> gauss(0.0, stddev_vector[i]);
        noise_generators.push_back(gauss);
    }
    for(auto& it: qoi_noisy){
        for(int j = 0; j < n_qois; ++j){
            double noise = noise_generators.at(j)(rng);
            it[j] += noise;
            if(cutoffZero){
                it[j] = std::max(0.0, it[j]);
            }
        }
    }
    return qoi_noisy;
}

std::vector<state_type> generate_data_sparse(const std::vector<state_type>& states, std::vector<double> stddev_vector, double fractionOfZeros){
    
    std::vector<state_type> data = generate_data(states, stddev_vector, true);
    size_t stateVectorSize = data.at(0).size();

    int k = int(data.size()*fractionOfZeros);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), gen);

    for (int i = 0; i < k; ++i) {
        data[indices[i]] = state_type(stateVectorSize,0);
    }

    return data;
}

void apply_qois(const state_type& in_state, const std::vector<std::vector<double>>& linear_combinations, state_type& out_state){
    // expects 0-initialized out_state
    for(int i=0; i<in_state.size(); ++i){
        for(int j=0; j<linear_combinations.size(); ++j){
            out_state[j] += linear_combinations[j][i] * in_state[i];
        }
    }
}

void FullModelEvaluator::evaluate(const state_type&x, state_type& dxdt, const double t, std::vector<double>& cache){
    double one_over_lf_M_150;
    double div;
    if(cache.at(parameterIndex(SEIRModel::ParamName::lf_M))<0.01){
        div = 0.01*150;
    }
    else{
        div = cache.at(parameterIndex(SEIRModel::ParamName::lf_M))*150;
    }
    one_over_lf_M_150 = 1/div;
    dxdt[0] = cache.at(parameterIndex(SEIRModel::ParamName::a))*cache.at(parameterIndex(SEIRModel::ParamName::beta_er))*(x[2]+x[3]+x[4]) - cache.at(parameterIndex(SEIRModel::ParamName::delta_E)) * x[0] - cache.at(parameterIndex(SEIRModel::ParamName::mu_E)) * x[0];
    dxdt[1] = cache.at(parameterIndex(SEIRModel::ParamName::delta_E)) * x[0] - cache.at(parameterIndex(SEIRModel::ParamName::delta_J)) * x[1] - cache.at(parameterIndex(SEIRModel::ParamName::alpha))*std::pow(x[1], 2) - cache.at(parameterIndex(SEIRModel::ParamName::mu_J)) * x[1];
    dxdt[2] = cache.at(parameterIndex(SEIRModel::ParamName::omega))*cache.at(parameterIndex(SEIRModel::ParamName::delta_J)) * x[1] - cache.at(parameterIndex(SEIRModel::ParamName::a))*cache.at(parameterIndex(SEIRModel::ParamName::b_M))*x[2]*x[7]/(x[5]+x[6]+x[7]+x[8]) - one_over_lf_M_150*x[2];
    dxdt[3] = cache.at(parameterIndex(SEIRModel::ParamName::a))*cache.at(parameterIndex(SEIRModel::ParamName::b_M))*x[2]*x[7]/(x[5]+x[6]+x[7]+x[8]) - 1/cache.at(parameterIndex(SEIRModel::ParamName::EIP_M))*x[3] - one_over_lf_M_150*x[3];
    dxdt[4] = 1/cache.at(parameterIndex(SEIRModel::ParamName::EIP_M))*x[3] - one_over_lf_M_150*x[4];
    dxdt[5] = cache.at(parameterIndex(SEIRModel::ParamName::Lambda)) - cache.at(parameterIndex(SEIRModel::ParamName::a))*cache.at(parameterIndex(SEIRModel::ParamName::b_H))*x[5]*x[4]/(x[2]+x[3]+x[4]) - cache.at(parameterIndex(SEIRModel::ParamName::mu_H))*x[5];
    dxdt[6] = cache.at(parameterIndex(SEIRModel::ParamName::a))*cache.at(parameterIndex(SEIRModel::ParamName::b_H))*x[5]*x[4]/(x[2]+x[3]+x[4]) - cache.at(parameterIndex(SEIRModel::ParamName::alpha_H))*x[6] - cache.at(parameterIndex(SEIRModel::ParamName::mu_H))*x[6];
    dxdt[7] = cache.at(parameterIndex(SEIRModel::ParamName::alpha_H))*x[6] - cache.at(parameterIndex(SEIRModel::ParamName::gamma_H))*x[7] - cache.at(parameterIndex(SEIRModel::ParamName::mu_H))*x[7];
    dxdt[8] = cache.at(parameterIndex(SEIRModel::ParamName::gamma_H))*x[7] - cache.at(parameterIndex(SEIRModel::ParamName::mu_H)) * x[8];
}

// void MosquitoPopulationEvaluator::evaluate(const state_type&x, 
//                                             state_type& dxdt, 
//                                             const double t, 
//                                             std::vector<double>& cache){
    
//     double one_over_lf_M_150;
//     double div;
//     if(cache[SEIRModel::ParamName::lf_M]<0.01){
//         div = 0.01*150;
//     }
//     else{
//         div = cache[SEIRModel::ParamName::lf_M]*150;
//     }
//     one_over_lf_M_150 = 1/div;
//     dxdt[0] = cache[SEIRModel::ParamName::a]*cache[SEIRModel::ParamName::beta_er]*(x[2]+x[3]+x[4]) - cache[SEIRModel::ParamName::delta_E] * x[0] - cache[SEIRModel::ParamName::mu_E] * x[0];
//     dxdt[1] = cache[SEIRModel::ParamName::delta_E] * x[0] - cache[SEIRModel::ParamName::delta_J] * x[1] - cache[SEIRModel::ParamName::alpha]*std::pow(x[1], 2) - cache[SEIRModel::ParamName::mu_J] * x[1];
//     dxdt[2] = cache[SEIRModel::ParamName::omega]*cache[SEIRModel::ParamName::delta_J] * x[1]  - one_over_lf_M_150*x[2];
// }

SEIRModel::SEIRModel(){
    m_cacheUpdateTolerance = 1;
    m_cachedTime = 0.0;
    temperature_profile = [](double t){
        return 15+15*std::sin(2*std::numbers::pi/360 * t);
    };
    // temperature_profile = [](double t){
    //     return 10+15*std::sin(2*std::numbers::pi/360 * t);
    // };
    m_indexedCache.resize(int(ParamName::count));
    ModelParameterFunctions.at(ParamName::delta_E) = fct::Briere(scaling_delta_E);
    ModelParameterFunctions.at(ParamName::p_E) = fct::Quadratic(scaling_p_E);
    ModelParameterFunctions.at(ParamName::mu_E) = fct::Pol4(scaling_mu_E);
    ModelParameterFunctions.at(ParamName::beta_er) = fct::Constant();
    ModelParameterFunctions.at(ParamName::alpha) = fct::Constant();
    ModelParameterFunctions.at(ParamName::delta_J) = fct::Briere(scaling_delta_J);
    ModelParameterFunctions.at(ParamName::p_J) = fct::Quadratic(scaling_p_J);
    ModelParameterFunctions.at(ParamName::mu_J) = fct::Pol4(scaling_mu_J);
    ModelParameterFunctions.at(ParamName::omega) = fct::Constant();
    ModelParameterFunctions.at(ParamName::lf_M) = fct::TruncatedLinear(scaling_lf_M);
    ModelParameterFunctions.at(ParamName::a) = fct::Briere(scaling_a);
    ModelParameterFunctions.at(ParamName::b_M) = fct::Sigmoidal();
    ModelParameterFunctions.at(ParamName::EIP_M) = fct::ExponentialDecay();
    ModelParameterFunctions.at(ParamName::Lambda) = fct::Constant();
    ModelParameterFunctions.at(ParamName::b_H) = fct::Constant();
    ModelParameterFunctions.at(ParamName::mu_H) = fct::Constant();
    ModelParameterFunctions.at(ParamName::alpha_H) = fct::Constant();
    ModelParameterFunctions.at(ParamName::gamma_H) = fct::Constant();
    updateCachedParameters(m_cachedTime, ModelParameters);
    m_evaluator = std::make_shared<FullModelEvaluator>();
}

SEIRModel SEIRModel::clone() const{
    SEIRModel newModel;
    newModel.temperature_profile = temperature_profile;
    newModel.m_evaluator = m_evaluator;
    for(const auto& param : SEIRModel::ModelParameters){
        newModel.set_function_parameter(param, get_function_param(param));
    }
    newModel.set_cacheUpdateTolerance(m_cacheUpdateTolerance);
    return newModel;
};

void SEIRModel::eval(state_type &x, state_type &dxdt, const double t ){
    // 0:E, 1:J, 2:S_M, 3:E_M, 4:I_M, 5:S_H, 6:E_H, 7:I_H, 8:R_H
    // beta = a*beta_ER
    if(m_evaluator == nullptr){
        spdlog::error("SEIRModel:: no evaluator");
    }
    if(std::abs(t - m_cachedTime) > m_cacheUpdateTolerance){
        m_cachedTime = t;
        updateCachedParameters(t, ModelParameters);
    }
    m_evaluator->evaluate(x, dxdt, t, m_indexedCache);
    // double one_over_lf_M_150;
    // double div;
    // if(m_cachedParameters[SEIRModel::ParamName::lf_M]<0.01){
    //     div = 0.01*150;
    // }
    // else{
    //     div = m_cachedParameters[SEIRModel::ParamName::lf_M]*150;
    // }
    // one_over_lf_M_150 = 1/div;
    // dxdt[0] = m_cachedParameters[SEIRModel::ParamName::a]*m_cachedParameters[SEIRModel::ParamName::beta_er]*(x[2]+x[3]+x[4]) - m_cachedParameters[SEIRModel::ParamName::delta_E] * x[0] - m_cachedParameters[SEIRModel::ParamName::mu_E] * x[0];
    // dxdt[1] = m_cachedParameters[SEIRModel::ParamName::delta_E] * x[0] - m_cachedParameters[SEIRModel::ParamName::delta_J] * x[1] - m_cachedParameters[SEIRModel::ParamName::alpha]*std::pow(x[1], 2) - m_cachedParameters[SEIRModel::ParamName::mu_J] * x[1];
    // dxdt[2] = m_cachedParameters[SEIRModel::ParamName::omega]*m_cachedParameters[SEIRModel::ParamName::delta_J] * x[1] - m_cachedParameters[SEIRModel::ParamName::a]*m_cachedParameters[SEIRModel::ParamName::b_M]*x[2]*x[7]/(x[5]+x[6]+x[7]+x[8]) - one_over_lf_M_150*x[2];
    // dxdt[3] = m_cachedParameters[SEIRModel::ParamName::a]*m_cachedParameters[SEIRModel::ParamName::b_M]*x[2]*x[7]/(x[5]+x[6]+x[7]+x[8]) - 1/m_cachedParameters[SEIRModel::ParamName::EIP_M]*x[3] - one_over_lf_M_150*x[3];
    // dxdt[4] = 1/m_cachedParameters[SEIRModel::ParamName::EIP_M]*x[3] - one_over_lf_M_150*x[4];
    // dxdt[5] = m_cachedParameters[SEIRModel::ParamName::Lambda] - m_cachedParameters[SEIRModel::ParamName::a]*m_cachedParameters[SEIRModel::ParamName::b_H]*x[5]*x[4]/(x[2]+x[3]+x[4]) - m_cachedParameters[SEIRModel::ParamName::mu_H]*x[5];
    // dxdt[6] = m_cachedParameters[SEIRModel::ParamName::a]*m_cachedParameters[SEIRModel::ParamName::b_H]*x[5]*x[4]/(x[2]+x[3]+x[4]) - m_cachedParameters[SEIRModel::ParamName::alpha_H]*x[6] - m_cachedParameters[SEIRModel::ParamName::mu_H]*x[6];
    // dxdt[7] = m_cachedParameters[SEIRModel::ParamName::alpha_H]*x[6] - m_cachedParameters[SEIRModel::ParamName::gamma_H]*x[7] - m_cachedParameters[SEIRModel::ParamName::mu_H]*x[7];
    // dxdt[8] = m_cachedParameters[SEIRModel::ParamName::gamma_H]*x[7] - m_cachedParameters[SEIRModel::ParamName::mu_H] * x[8];
};
    
void SEIRModel::updateCachedParameters(double t, const std::vector<ParamName>& whichOnes){
    m_cachedTemperature = temperature_profile(t); 
    for(auto& updParameter:whichOnes){
        // double newVal = ModelParameterFunctions.at(updParameter)(m_cachedTemperature, get_function_param(updParameter));
        m_indexedCache.at(parameterIndex(updParameter)) = ModelParameterFunctions.at(updParameter)(m_cachedTemperature, get_function_param(updParameter));;
    }
};

std::vector<double>& SEIRModel::get_function_param(ParamName name){
    switch(name){
        case ParamName::delta_E: return fp_delta_E;
        case ParamName::p_E: return fp_p_E;
        case ParamName::mu_E: return fp_mu_E;
        case ParamName::alpha: return fp_alpha;
        case ParamName::delta_J: return fp_delta_J;
        case ParamName::p_J: return fp_p_J;
        case ParamName::mu_J: return fp_mu_J;
        case ParamName::omega: return fp_omega;
        case ParamName::lf_M: return fp_lf_M;
        case ParamName::a: return fp_a;
        case ParamName::beta_er: return fp_beta_er;
        case ParamName::b_M: return fp_b_M;
        case ParamName::EIP_M: return fp_EIP_M;
        case ParamName::Lambda: return fp_Lambda;
        case ParamName::b_H: return fp_b_H;
        case ParamName::mu_H: return fp_mu_H;
        case ParamName::alpha_H: return fp_alpha_H;
        case ParamName::gamma_H: return fp_gamma_H;
        default: throw std::out_of_range("invalid parameter");
    }
}

std::vector<double> SEIRModel::get_function_param(ParamName name) const {
    switch(name){
        case ParamName::delta_E: return fp_delta_E;
        case ParamName::p_E: return fp_p_E;
        case ParamName::mu_E: return fp_mu_E;
        case ParamName::alpha: return fp_alpha;
        case ParamName::delta_J: return fp_delta_J;
        case ParamName::p_J: return fp_p_J;
        case ParamName::mu_J: return fp_mu_J;
        case ParamName::omega: return fp_omega;
        case ParamName::lf_M: return fp_lf_M;
        case ParamName::a: return fp_a;
        case ParamName::beta_er: return fp_beta_er;
        case ParamName::b_M: return fp_b_M;
        case ParamName::EIP_M: return fp_EIP_M;
        case ParamName::Lambda: return fp_Lambda;
        case ParamName::b_H: return fp_b_H;
        case ParamName::mu_H: return fp_mu_H;
        case ParamName::alpha_H: return fp_alpha_H;
        case ParamName::gamma_H: return fp_gamma_H;
        default: throw std::out_of_range("invalid parameter");
    }
};
    
const std::map<SEIRModel::ParamName, seir_param> SEIRModel::getAllFunctions() const {
    return ModelParameterFunctions;
};
const std::vector<std::vector<double>> SEIRModel::getAllFunctionParameters() const{
    spdlog::warn("SEIRModel::getAllFunctionParameters:: This is not safe, they might be out of order");
    std::vector<std::vector<double>> result{
        fp_delta_E,
        fp_p_E,
        fp_mu_E,
        fp_alpha,
        fp_delta_J,
        fp_p_J,
        fp_mu_J,
        fp_omega,
        fp_lf_M,
        fp_a,
        fp_beta_er,
        fp_b_M,
        fp_EIP_M,
        fp_Lambda,
        fp_b_H,
        fp_mu_H,
        fp_alpha_H,
        fp_gamma_H
    };
    return result;
};

void SEIRModel::set_function_parameter(ParamName name, const std::vector<double> &value){
    size_t paramSize = get_function_param(name).size();
    if(paramSize != value.size()){
        throw std::runtime_error("Wrong size to set parameter");
    }
    get_function_param(name) = value;
}

SEIRObserver::SEIRObserver( std::vector< state_type > &states , std::vector< double > &times): m_states(states), m_times(times){};

void SEIRObserver::writeResult(std::string path){
    std::ofstream solution_file(path);
    if (!solution_file.is_open()) {
        std::cout<<"Failed to save file"<<std::endl;
        exit(-1);
    }
    size_t state_size = m_states[0].size();
    for(int i=0; i<m_states.size(); ++i){
        solution_file<<m_times[i]<<",";
        for(int j = 0; j<state_size; ++j){
            solution_file<<m_states[i][j];
            if(j<state_size-1){
                solution_file<<",";
            }
        }
        solution_file<<std::endl;;
    }
}

void SEIRObserver::add_qoi(const std::vector<double> &lc_qoi){
        if(m_current_index > 0){
            std::cout<<"Warn: observation has started, qoi cannot be changed anymore"<<std::endl;;
            return;
        }
        m_qoi_linear_combinations.push_back(lc_qoi);
    }

void SEIRObserver::observe_qois( const state_type &x, double t){
    apply_qois(x, m_qoi_linear_combinations, m_states.at(m_current_index));
    m_times.at(m_current_index) = t;
    ++m_current_index;
}

void SEIRObserver::observe_full( const state_type &x, double t){
        m_states.at(m_current_index) = x;
        m_times.at(m_current_index) = t;
        ++m_current_index;
    }

size_t SEIRObserver::number_of_quantities(){
    size_t n_quantities = m_qoi_linear_combinations.size();
    if(n_quantities == 0){
        n_quantities = m_states[0].size();
    }
    return n_quantities;
}

void SEIRObserver::restart(){
    size_t n_states = m_times.size();
    size_t n_quantities = number_of_quantities();
    state_type zero_state(n_quantities, 0.0);
    std::fill(m_states.begin(), m_states.end(), zero_state);
    std::fill(m_times.begin(), m_times.end(), 0.0);
    m_current_index = 0;
}

solverSettings::solverSettings(double start, double end, double stepsize, state_type initialState):startTime(start), endTime(end), stepSize(stepsize), initialState(initialState){
        dim = initialState.size();
        nSteps = int(endTime/stepSize);
        solverGrid = make_grid(startTime, endTime, nSteps);
    }

void ParameterDescription::validate(){
    assert(m_totalSize >= m_subsetIndices.size());
    for(const auto& index:m_subsetIndices){
        assert(index < m_totalSize);
    }
}

size_t ParameterDescription::size() const {
    if(!m_subsetIndices.empty()){ return m_subsetIndices.size(); }
    return m_totalSize;
}

bool ParameterDescription::isSubset(const ParameterDescription& supersetDesc) const{
    if(m_name != supersetDesc.m_name){return false;}
    if(supersetDesc.size() == supersetDesc.m_totalSize){return true;}
    auto supersetIndices = supersetDesc.getSubsetIndices();
    auto selfIndices = getSubsetIndices();
    for(const auto& subIdx: selfIndices){
        auto where = std::find(supersetIndices.begin(), supersetIndices.end(), subIdx);
        if(where == supersetIndices.end()){
            return false;
        }
    }
    return true;
}

bool ParameterDescription::isSuperset(const ParameterDescription& subsetDesc) const{
    return subsetDesc.isSubset(*this);
}

std::set<int> ParameterDescription::getSubsetIndices() const {
    std::vector<int> indexVec;
    if(size() == m_totalSize){
        indexVec.resize(m_totalSize);
        std::iota(indexVec.begin(), indexVec.end(), 0);
    }
    else{
        indexVec = m_subsetIndices;
    }
    std::set<int> indexSet = std::set<int>(indexVec.begin(), indexVec.end());
    return indexSet;
}

bool operator==(const ParameterDescription& desc1, const ParameterDescription& desc2){
    return desc1.isSubset(desc2) && desc2.isSubset(desc1);
};

std::vector<double> make_grid(double startTime, double endTime, size_t n_measurements){
    double increment = (endTime-startTime)/(n_measurements-1);
    std::vector<double> measurement_times(n_measurements, startTime);
    measurement_times[n_measurements-1] = endTime;
    for(int i = 1; i<n_measurements-1; ++i){
        measurement_times[i] = measurement_times[i-1] + increment;
    }
    return measurement_times;
};

std::vector< state_type > odeSolve(SEIRModel& model, std::vector<double> measurementTimes, std::vector<double> initialValue, double stepsize, std::optional<std::string> writeCsv){
    double startTime = measurementTimes.at(0);
    double endTime = measurementTimes.back();
    int n_measurements = measurementTimes.size();
    std::vector<double> solver_grid = make_grid(startTime, endTime, int(endTime/stepsize));
    boost::numeric::odeint::runge_kutta4 < state_type > stepper;
    std::vector< state_type > states_vec{measurementTimes.size(), std::vector<double>(initialValue.size(), 0.0)};
    // measures_vec{n_measurements, std::vector<double>(numberQois, 0.0)};
    std::vector< double > times(solver_grid.size());
    state_type odeState = initialValue;
    SEIRObserver observer(states_vec, measurementTimes);
    auto meas_time_iterator = measurementTimes.begin();
    state_type prev_state;
    bool write_meas_flag = false;
    auto start = std::chrono::high_resolution_clock::now();
    for(const auto& time: solver_grid){
        if(time + stepsize > *meas_time_iterator){
            prev_state = odeState;
            write_meas_flag = true;
        }
        // observer.observe_full(odeState, time);
        auto rhs_eval = [&model](state_type &x, state_type &dxdt, const double t ){model.eval(x, dxdt, t);};
        stepper.do_step(rhs_eval, odeState, time, stepsize);
        // std::for_each(odeState.begin(), odeState.end(), [](double& val){val = std::max(0.0, val);});
        if(write_meas_flag){
            while(*meas_time_iterator<time + stepsize && meas_time_iterator<measurementTimes.end()){
                state_type intermediate_state(odeState.size(), 0.0);
                for(int i = 0; i < odeState.size(); ++i){
                    intermediate_state[i] = prev_state[i] + (*meas_time_iterator-time)*(odeState[i] - prev_state[i])/(stepsize);
                }
                observer.observe_full(odeState, time);
                // measureObserver.observe_qois(intermediate_state, *meas_time_iterator);
                ++meas_time_iterator;
            }
            write_meas_flag = false;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if(writeCsv.has_value()){
        std::filesystem::path out = writeCsv.value();
        if(!std::filesystem::is_directory(out.parent_path())){std::filesystem::create_directories(out.parent_path());}
        observer.writeResult(writeCsv.value());
    }
    return states_vec;
};

void evaluateModelParameters(const SEIRModel& model, std::string outFolder){
    auto paramFunctions = model.getAllFunctions();
    auto functionParameters = model.getAllFunctionParameters();
    auto grid = make_grid(0,80,1000);
    std::vector<std::vector<double>> xy;
    xy.push_back(grid);
    // for(int i=0; i<paramFunctions.size(); ++i){
    int mapidx = -1;
    for(const auto& [key,boostfunc]:paramFunctions){
        mapidx ++;
        auto funcParam = functionParameters.at(mapidx);
        auto stdfunc = [boostfunc, funcParam](double x){return boostfunc(x, funcParam);};
        auto evals = plot::evaluate1DFunction<double>(grid, stdfunc);
        xy.push_back(evals);
    }
    std::vector<std::vector<double>> xyt;
    
    for( int jj = 0; jj<xy[0].size(); ++jj){
        std::vector<double> newRow;
        for(int ii = 0 ; ii<xy.size(); ++ii){
            newRow.push_back(xy.at(ii).at(jj));
        }
        xyt.push_back(newRow);
    }
    
    writeCSV(xyt, outFolder + "/seirEvals.csv");
};

std::vector<double> calculateR0(SEIRModel& model, solverSettings settings){
    std::vector<state_type> odeSolution = odeSolve(model, settings.solverGrid, settings.initialState, settings.stepSize);
    auto parameterFunctions = model.getAllFunctions();
    auto temperatureProfile = model.get_temperature_profile();
    auto a_func = [&parameterFunctions, &temperatureProfile, &model](double t){
        return parameterFunctions.at(SEIRModel::ParamName::a)(temperatureProfile(t), model.get_function_param(SEIRModel::ParamName::a));
    };
    auto b_M_func = [&parameterFunctions, &temperatureProfile, &model](double t){
        return parameterFunctions.at(SEIRModel::ParamName::b_M)(temperatureProfile(t), model.get_function_param(SEIRModel::ParamName::b_M));
    };
    auto b_H_func = [&parameterFunctions, &temperatureProfile, &model](double t){
        return parameterFunctions.at(SEIRModel::ParamName::b_H)(temperatureProfile(t), model.get_function_param(SEIRModel::ParamName::b_H));
    };
    auto lf_M_func = [&parameterFunctions, &temperatureProfile, &model](double t){
        return parameterFunctions.at(SEIRModel::ParamName::lf_M)(temperatureProfile(t), model.get_function_param(SEIRModel::ParamName::lf_M));
    };
    auto EIP_M_func = [&parameterFunctions, &temperatureProfile, &model](double t){
        return parameterFunctions.at(SEIRModel::ParamName::EIP_M)(temperatureProfile(t), model.get_function_param(SEIRModel::ParamName::EIP_M));
    };
    auto gamma_H_func = [&parameterFunctions, &temperatureProfile, &model](double t){
        return parameterFunctions.at(SEIRModel::ParamName::gamma_H)(temperatureProfile(t), model.get_function_param(SEIRModel::ParamName::gamma_H));
    };
    std::vector<double> R0_vec;
    for(int i = 0; i < settings.solverGrid.size(); ++i){
        double t = settings.solverGrid.at(i);
        double mosqHostRatio = (odeSolution.at(i).at(2)+odeSolution.at(i).at(3)+odeSolution.at(i).at(4))/((odeSolution.at(i).at(5)+odeSolution.at(i).at(6)+odeSolution.at(i).at(7)+odeSolution.at(i).at(8)));
        double R0 = mosqHostRatio*150*lf_M_func(t)/gamma_H_func(t)*std::pow(a_func(t),2)*b_M_func(t)*b_H_func(t)*std::exp(-1.0*EIP_M_func(t)/(150*lf_M_func(t)));
        R0_vec.push_back(R0);
    }
    return R0_vec;
}