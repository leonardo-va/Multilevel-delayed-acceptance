#include <sampler.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#include <memory>
#include <magic_enum/magic_enum.hpp>
#include <io.h>

std::unique_ptr<indicators::ProgressBar> makeProgressBar(int maxSteps, std::string name){
    std::unique_ptr<indicators::ProgressBar> newBar = std::make_unique<indicators::ProgressBar>(
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{" ]"},
        indicators::option::ForegroundColor{indicators::Color::blue},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::PrefixText{name},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
        indicators::option::MaxProgress{maxSteps}
    );
    return newBar;
}

void Sampler::add_inferred_param(ParameterDescription paramDescr, const multiNormalIndependentDist& priorDens){
    if(paramDescr.size() != priorDens.getDim()){
        throw std::invalid_argument("add inferred parameter:: wrong parameter size");
    }
    for(const auto& inferred : m_stateDescription){
        if(inferred.m_name == paramDescr.m_name){
            spdlog::warn("Sampler {}:: failed to add inferred parameter, it already exists", name);
            return;
        }
    }
    m_stateDescription.push_back(paramDescr);
    if(!m_prior){
        m_prior = std::make_shared<multiNormalIndependentDist>();
    }
    auto priorCast = std::dynamic_pointer_cast<multiNormalIndependentDist>(m_prior);
    m_prior = std::make_shared<multiNormalIndependentDist>(*priorCast * priorDens);
    // m_prior = &((*dynamic_cast<multiNormalIndependentDist>) * priorDens);
}

void Sampler::add_inferred_param(ParameterDescription& paramDescr){
    for(const auto& inferred : m_stateDescription){
        if(inferred.m_name == paramDescr.m_name){
            spdlog::warn("Sampler {}:: failed to add inferred parameter, it already exists", name);
            return;
        }
    }
    m_stateDescription.push_back(paramDescr);
}

double Sampler::log_prior_eval(const std::vector<parameter_type>& theta){
    // need a test for this
    auto start_prior = std::chrono::high_resolution_clock::now();
    if(theta.size() != m_stateDescription.size()){
        spdlog::error("in sampler {}, theta size: {}, inferred list size: {}", name, theta.size(), m_stateDescription.size());
        throw std::invalid_argument("Sampler:: size error evaluating log prior");
    }

    std::vector<double> flattenedTheta(m_prior->getDim(), 0.0);
    auto flat_it = flattenedTheta.begin();
    int current = 0;
    for(const auto& it: theta){
        current += it.size();
        if(current > flattenedTheta.size()){throw std::out_of_range("Sampler:: input theta has wrong dimension for prior eval");};
        std::copy(it.begin(), it.end(), flat_it);
        std::advance(flat_it, it.size());
    }
    double retval = log_prior_eval(flattenedTheta);
    auto end_prior = std::chrono::high_resolution_clock::now();
    auto duration_prior = std::chrono::duration_cast<std::chrono::microseconds>(
        end_prior - start_prior
    );
    m_timings.latest().logPrior = duration_prior;
    return retval;
}

double Sampler::log_prior_eval(parameter_type theta){
    double retval;
    try { retval = m_prior->logpdf(theta);}
    catch(std::invalid_argument e){
        std::string throw_msg = "Sampler:: prior evaluation failed with " + std::string(e.what());
        throw std::invalid_argument(throw_msg);
    }
    m_cachedLogPrior = retval;
    return retval;
}

double Sampler::log_likelihood(const std::vector< parameter_type >& parameter){
    // check again if this is correct(likelihood)
    auto start_likelihood = std::chrono::high_resolution_clock::now();
    forward(parameter);
    double log_likelihood = 0;
    for(int i=0; i<m_data.size(); ++i){
        // auto dens = multiNormalIndependentDist(m_forwardObsStates[i], m_forwardObsSigmaSquared);
        auto dens = multiNormalDist(m_forwardObsStates[i], m_forwardObsCovMat);
        log_likelihood += dens.logpdf(state_type(m_data[i].begin()+1, m_data[i].end()));
    }
    m_cachedLogLikelihood = log_likelihood;
    auto end_likelihood = std::chrono::high_resolution_clock::now();
    auto duration_likelihood = std::chrono::duration_cast<std::chrono::microseconds>(
        end_likelihood - start_likelihood
    );
    m_timings.latest().logLikelihood = duration_likelihood;
    return log_likelihood;
};

/// @brief This calculates log posterior of the given state, and updates the internal cache values for logposterior,
/// logprior and loglikelihood
/// @param parameter 
/// @return 
double Sampler::log_posterior_eval(const std::vector<parameter_type>& parameter){
    // check if the proposal is the same as the current state, and skip the likelihood evaluation if this is the case
    bool noNewState = true;
    auto currentState = getCurrentState();
    double newStateTolerance = 1e-12;
    for(int ii = 0; ii<parameter.size(); ++ii){
        if(!noNewState){break;}
        for(int jj=0; jj<parameter.at(ii).size(); ++jj){
            if(std::abs(parameter.at(ii).at(jj) - currentState.at(ii).at(jj)) > newStateTolerance){
                noNewState = false;
                break;
            }
        }
    }

    // if new state or if first state make the evaluation. First state is important because the initialization calls this after setting the initial state,
    // so it will detect a noNewState and keep garbage in cachedLogPosterior/Prior/Likelihood otherwise
    if((!noNewState) || m_stepCounter == 0){
        m_cachedLogPosterior = log_prior_eval(parameter) + log_likelihood(parameter);
    }
    else{
        // use this instead of cache in case cache holds something weird,
        // also overwrite the cache for prior and likelihood as that is supposed to happen on this eval call
        m_cachedLogPosterior = getLogPosteriorValues().back();
        m_cachedLogLikelihood = getLogLikelihoodValues().back();
        m_cachedLogPrior = getLogPriorValues().back();
    }
    return m_cachedLogPosterior;
};

void Sampler::registerModel(SEIRModel model){m_model = model;}

void Sampler::addData(const std::vector< state_type >& data){
    // expect format vector of [time, obs1, obs2, ...]
    spdlog::info("Sampler {}:: adding data: {} measurements of {} quantities", name, data.size(), data[0].size()-1);
    m_data = data;
    m_measure_times.clear();
    m_forwardObsStates.clear();
    m_measure_times.resize(data.size());
    m_numberQois = data[0].size();
    m_forwardObsStates.resize(data.size(), state_type(m_numberQois, 0.0));
    m_forwardObsTimes = m_measure_times;
    for(int i = 0; i<data.size(); ++i){
        m_measure_times.at(i) = data[i].at(0);
    }
    m_forwardObserver.emplace(SEIRObserver(m_forwardObsStates, m_forwardObsTimes));
};

void Sampler::set_solverSettings(const solverSettings& settings){m_solverSettings.emplace(settings);};

void Sampler::add_observed_quantity(const std::vector<double> qoi_lc, double stddev){
    // add in order of the data columns
    if(m_tempered){
        spdlog::warn("sampler {} :: add_observed_quantity: Likelihood is tempered, cannot add new data");
    }

    if(!m_forwardObserver.has_value()){
        std::cerr<<"Sampler:: failed to add observed quantity, no data has been registered"<<std::endl;;
        return;
    }
    int qoi_index = m_forwardObserver->m_qoi_linear_combinations.size() - 1;
    if(m_forwardObserver->m_qoi_linear_combinations.size() >= m_numberQois){
        std::cerr<<"Sampler:: not possible to add more quantities"<<std::endl;;
        return;
    }
    m_forwardObserver.value().add_qoi(qoi_lc);
    m_forwardObsSigmaSquared.push_back(stddev*stddev);
    std::vector<double> newRow(m_forwardObsCovMat.size() + 1, 0.0);
    newRow.back() = stddev*stddev;
    for(auto& row:m_forwardObsCovMat){
        row.push_back(0.0);
    }
    m_forwardObsCovMat.push_back(newRow);
};

// this should only be called once i think at the beginning. Probably not safe to reuse a sampler that has been
// running already because of step counter stuff (first posterior eval will likely fail). It also deletes the history file on disk
void Sampler::set_initialState(const std::vector< parameter_type >& initialState){
    clearHistory();
    // m_stepCounter = 0;
    m_chain.push_back(initialState);
    m_proposalHistory.push_back(initialState);
    log_posterior_eval(initialState);
    m_currentStateLogPosterior = m_cachedLogPosterior;
    m_currentStateLogPrior = m_cachedLogPrior;
    m_currentStateLogLikelihood = m_cachedLogLikelihood;
    m_acceptanceProbs.push_back(1.);
    m_accepted.push_back(1);
    m_logposteriorVals.push_back(m_cachedLogPosterior);
    m_logpriorVals.push_back(m_cachedLogPrior);
    m_loglikelihoodVals.push_back(m_cachedLogLikelihood);
    m_forced.push_back(1);
    // create output file with header, later states are only appended. Ignore the initial state, it is written with the first dump
    writeSamplerOutput(this, m_outFolderPath, false, "w", 1);
};

// set the next state by force, dont call this on normal step
void Sampler::set_nextState(const std::vector< parameter_type >& newState){
    ++m_stepCounter;
    if(m_stepCounter<=m_burnin+1){
        m_chain.at(0)=newState;
        m_proposalHistory.at(0)=newState;
        log_posterior_eval(newState);
        m_currentStateLogPosterior = m_cachedLogPosterior;
        m_currentStateLogPrior = m_cachedLogPrior;
        m_currentStateLogLikelihood = m_cachedLogLikelihood;
        m_acceptanceProbs.at(0)=1.;
        m_accepted.at(0)=1;
        m_logposteriorVals.at(0)=m_cachedLogPosterior;
        m_logpriorVals.at(0)=m_cachedLogPrior;
        m_loglikelihoodVals.at(0)=m_cachedLogLikelihood;
        m_forced.at(0)=1;
    }
    else{
        m_chain.push_back(newState);
        m_proposalHistory.push_back(newState);
        log_posterior_eval(newState);
        m_currentStateLogPosterior = m_cachedLogPosterior;
        m_currentStateLogPrior = m_cachedLogPrior;
        m_currentStateLogLikelihood = m_cachedLogLikelihood;
        m_acceptanceProbs.push_back(1.);
        m_accepted.push_back(1);
        m_logposteriorVals.push_back(m_cachedLogPosterior);
        m_logpriorVals.push_back(m_cachedLogPrior);
        m_loglikelihoodVals.push_back(m_cachedLogLikelihood);
        m_forced.push_back(1);
    }
};

void Sampler::set_minimumHistorySize(uint minSize){
    if(minSize>=m_maximumChainSize){
        spdlog::warn("Sampler {} :: failed to set minimum history size {}", name,minSize);
        return;
    }
    else{
        m_keepOnErase = minSize;
    }
}

void Sampler::clearHistory(){
    m_chain.clear();
    m_proposalHistory.clear();
    m_acceptanceProbs.clear();
    m_logpriorVals.clear();
    m_loglikelihoodVals.clear();
    m_logposteriorVals.clear();
    m_accepted.clear();
    m_forced.clear();
   
    m_cachedLogPosterior = std::numeric_limits<double>::quiet_NaN();
    m_cachedLogPrior = std::numeric_limits<double>::quiet_NaN();
    m_cachedLogLikelihood = std::numeric_limits<double>::quiet_NaN();
    m_currentStateLogPosterior = std::numeric_limits<double>::quiet_NaN();
    m_currentStateLogPrior = std::numeric_limits<double>::quiet_NaN();
    m_currentStateLogLikelihood = std::numeric_limits<double>::quiet_NaN();
    m_forwardObserver.value().restart();
};

void Sampler::eraseHistory(uint keepLastN){
    m_chain.erase(m_chain.begin(), m_chain.end() - keepLastN);
    m_proposalHistory.erase(m_proposalHistory.begin(), m_proposalHistory.end() - keepLastN);
    m_acceptanceProbs.erase(m_acceptanceProbs.begin(), m_acceptanceProbs.end() - keepLastN);
    m_logpriorVals.erase(m_logpriorVals.begin(), m_logpriorVals.end() - keepLastN);
    m_loglikelihoodVals.erase(m_loglikelihoodVals.begin(), m_loglikelihoodVals.end() - keepLastN);
    m_logposteriorVals.erase(m_logposteriorVals.begin(), m_logposteriorVals.end() - keepLastN);
    m_accepted.erase(m_accepted.begin(), m_accepted.end() - keepLastN);
    m_forced.erase(m_forced.begin(), m_forced.end() - keepLastN);
    if(m_chain.size() != keepLastN){
        spdlog::warn("Size after erasing history {}, keep Last N {}, size mismatch", m_chain.size(), keepLastN);
        throw std::runtime_error("Sampler:: erase history failed, size mismatch" + std::to_string(m_chain.size()) + " != " + std::to_string(keepLastN));
    } 
}

void Sampler::set_proposalMechanism(ProposalMechanismBase* propose){
    if(!propose){
        spdlog::error("sampler {}:: proposal mechanism is null", name);
        throw std::runtime_error("failed to set proposal mechanism");
    }
    m_proposalMechanism = propose;
    m_proposalMechanism->setOwningSampler(*this);
    auto finePtr = dynamic_cast<FineLevelProposal*>(propose);
    if(finePtr){
        level = finePtr->getCoarseSampler()->level + 1;
    }
};

void Sampler::forward(const std::vector< parameter_type > &proposed_parameter){
    auto start_forward = std::chrono::high_resolution_clock::now();
    if(m_stateDescription.size() != proposed_parameter.size()){throw std::invalid_argument("Sampler:: forward solve not possible, wrong parameter size");}
    for(int i=0; i<proposed_parameter.size(); ++i){
        if(proposed_parameter[i].size() != m_stateDescription[i].size()){throw std::invalid_argument("Sampler:: forward solve not possible, wrong parameter size");}
        if(m_stateDescription[i].size() == m_stateDescription[i].m_totalSize){
            m_model.set_function_parameter(m_stateDescription[i].m_name, proposed_parameter[i]);
        }
        else{
            auto modifiedFullParameter = m_model.get_function_param(m_stateDescription[i].m_name);
            for(int j=0; j<m_stateDescription[i].size(); ++j){
                modifiedFullParameter[m_stateDescription[i].m_subsetIndices[j]] = proposed_parameter[i][j];
            }
            m_model.set_function_parameter(m_stateDescription[i].m_name, modifiedFullParameter);
        }
    }
    if(!m_solverSettings.has_value()){throw std::runtime_error("Sampler:: forward solve not possible, solver settings have not been specified");}
    m_forwardObserver.value().restart();
    auto meas_time_iterator = m_measure_times.begin();
    state_type prev_state;
    state_type odeState = m_solverSettings->initialState;
    bool write_meas_flag = false;
    auto start = std::chrono::high_resolution_clock::now();
    for(const auto& time: m_solverSettings->solverGrid){
        if(time + m_solverSettings->stepSize > *meas_time_iterator){
            prev_state = odeState;
            write_meas_flag = true;
        }
        // observer.observe_full(odeState, time);
        auto rhs_eval = [this](state_type &x, state_type &dxdt, const double t ){m_model.eval(x, dxdt, t);};
        m_stepper.do_step(rhs_eval, odeState, time, m_solverSettings->stepSize);
        // std::for_each(odeState.begin(), odeState.end(), [](double& val){val = std::max(0.0, val);});
        if(write_meas_flag){
            while(*meas_time_iterator<time + m_solverSettings->stepSize && meas_time_iterator<m_measure_times.end()){
                state_type intermediate_state(odeState.size(), 0.0);
                for(int i = 0; i < odeState.size(); ++i){
                    intermediate_state[i] = prev_state[i] + (*meas_time_iterator-time)*(odeState[i] - prev_state[i])/(m_solverSettings->stepSize);
                }
                m_forwardObserver->observe_qois(intermediate_state, *meas_time_iterator);
                ++meas_time_iterator;
            }
            write_meas_flag = false;
        }
    }
    auto end_forward = std::chrono::high_resolution_clock::now();
    auto duration_forward = std::chrono::duration_cast<std::chrono::microseconds>(
        end_forward - start_forward
    );
    m_timings.latest().forward = duration_forward;
    // std::cout << "Sampler::" << name << "forward Elapsed time: " << elapsed.count() << " seconds\n";                    
}

void Sampler::set_prior(std::shared_ptr<distribution> priorDist){
    m_prior = priorDist;
};

void Sampler::set_burnin(uint burnin, bool removeFromHistory){
    m_burnin = burnin;
    m_removeBurninFromHistory = removeFromHistory;
};

void Sampler::set_outputFolder(std::string outFolderPath){ m_outFolderPath = outFolderPath; };

double Sampler::acceptanceProb(const std::vector< parameter_type >& parameter){
    double acceptance;
    try{
        acceptance = m_proposalMechanism->acceptanceProbability(this, parameter);
    }
    catch(const std::exception& e) {
        spdlog::error("{}, step {} :: failed to calculate acceptance probability", name, m_stepCounter);
        std::cout << e.what() << '\n';
        std::exit(1);
    }
    return acceptance;
};

void Sampler::step(){
    ++m_stepCounter;
    if(m_chain.size() >= m_maximumChainSize){
        dumpHistory();
    }
    m_timings.push(SamplerTiming{});
    auto start_propose = std::chrono::high_resolution_clock::now();
    std::vector< parameter_type > proposedState = (*m_proposalMechanism)(getCurrentState());
    auto end_propose = std::chrono::high_resolution_clock::now();
    auto duration_propose = std::chrono::duration_cast<std::chrono::microseconds>(
        end_propose - start_propose
    );
    m_timings.latest().proposal = duration_propose;
    // acceptance prob calls the proposal mechanisms function for calculating acceptance prob. This is then expected to 
    // call the samplers log posterior evaluation once, which updates the 3 cached values logpost, logprior, loglikelihood
    
    auto start_probAccept = std::chrono::high_resolution_clock::now();
    double probAccept = acceptanceProb(proposedState);
    auto end_probAccept = std::chrono::high_resolution_clock::now();
    auto duration_probAccept = std::chrono::duration_cast<std::chrono::microseconds>(
        end_probAccept - start_probAccept
    );
    m_timings.latest().acceptanceProbability = duration_probAccept;
    double unifDraw = m_uniform01(m_rngEngine);
    bool accepted = false;
    if(probAccept > unifDraw){
        // m_chain.push_back(proposedState);
        // m_accepted.push_back(1);
        accepted = true;
        m_currentStateLogPosterior = m_cachedLogPosterior;
        m_currentStateLogPrior = m_cachedLogPrior;
        m_currentStateLogLikelihood = m_cachedLogLikelihood;
    }
    auto start_updateHistory = std::chrono::high_resolution_clock::now();
    // update history
    if(m_stepCounter <= m_burnin+1){
        m_proposalHistory.at(0)=proposedState;
        if(accepted){
            m_chain.at(0)=proposedState;
            m_accepted.at(0)=1;
        }
        else{
            m_chain.at(0)=m_chain.back();
            m_accepted.at(0)=0;
        }
        m_acceptanceProbs.at(0)=probAccept;
        m_loglikelihoodVals.at(0)=m_currentStateLogLikelihood;
        m_logpriorVals.at(0)=m_currentStateLogPrior;
        m_logposteriorVals.at(0)=m_currentStateLogPosterior;
        m_forced.at(0)=0;
    }
    else{
        m_proposalHistory.push_back(proposedState);
        if(accepted){
            m_chain.push_back(proposedState);
            m_accepted.push_back(1);
        }
        else{
            m_chain.push_back(m_chain.back());
            m_accepted.push_back(0);
        }
        m_acceptanceProbs.push_back(probAccept);
        m_loglikelihoodVals.push_back(m_currentStateLogLikelihood);
        m_logpriorVals.push_back(m_currentStateLogPrior);
        m_logposteriorVals.push_back(m_currentStateLogPosterior);
        m_forced.push_back(0);
    }
    auto end_updateHistory = std::chrono::high_resolution_clock::now();
    auto duration_updateHistory = std::chrono::duration_cast<std::chrono::microseconds>(
        end_updateHistory - start_updateHistory
    );
    m_timings.latest().historyUpdate = duration_updateHistory;
}

void Sampler::dumpHistory(){
    if(m_outFolderPath.empty()){
        spdlog::warn("Sampler {}:: output folder is not set, history is lost", name);
        return;
    }
    writeSamplerOutput(this, m_outFolderPath, false, "a", m_keepOnErase);
    eraseHistory(m_keepOnErase);
};

/// @brief run chain for n_steps. n_steps needs to include any possible warmup samples, this function does not manage warmup.
/// @param n_steps 
/// @param noProgressBar Dont output the progressbar, necessary for subchains in multilevel sampler.
/// @param progressTicker For externally managed progress bars in multithreaded application.
void Sampler::run_for(size_t n_steps, bool noProgressBar, std::function<void()> progressTicker){
    if(m_proposalMechanism == nullptr){
        spdlog::error("Sampler {}: no proposal mechanism defined", name);
        return;
    }
    auto defaultBar = makeProgressBar(n_steps, "Chain #1 ("+name+")");
    if(!progressTicker){
        progressTicker = [&defaultBar](){defaultBar->tick();};
    }

    if(m_chain.size() < 1){spdlog::warn("Sampler::run_for:: Sampler is not initialized");}
    int stepCountLocal = 0;
    while(stepCountLocal < n_steps){
        ++stepCountLocal;
        if(!noProgressBar && progressTicker){
            progressTicker();
        }
        try{
            step();
        }
        catch(const std::exception& e) {
            spdlog::error("{} failed on step {}", name, m_stepCounter);
            std::cout << e.what() << '\n';
            std::exit(1);
        }
    }
}

const std::vector<parameter_type>& Sampler::getCurrentState() const{
    return m_chain.back();
};

double Sampler::getCache(std::string what) const {
    double retval = std::numeric_limits<double>::quiet_NaN();
    if(what=="logPosterior"){
        retval = m_cachedLogPosterior;
    }
    else if(what=="logPrior"){
        retval = m_cachedLogPrior;
    }
    else if(what=="logLikelihood"){
        retval = m_cachedLogLikelihood;
    }
    else{
        spdlog::warn("Sampler {}::getCache: unknown cache entry {}",name, what);
    }
    return retval;
};

const std::vector< std::vector< parameter_type > >& Sampler::getChain() const {
    return m_chain;
};

const std::vector< std::vector< parameter_type > >& Sampler::getProposalHistory() const {
    return m_proposalHistory;
};

const std::vector< double >& Sampler::getAcceptanceProbs() const {
    return m_acceptanceProbs;
};

const std::vector< double >& Sampler::getLogPriorValues() const{ 
    return m_logpriorVals;
};

const std::vector< double >& Sampler::getLogLikelihoodValues() const { 
    return m_loglikelihoodVals;
};

const std::vector< double >& Sampler::getLogPosteriorValues() const {
    return m_logposteriorVals;
};

/// @brief return the acceptance history
/// @return vector of int, -1: forced state (initial or set manually), 0: rejected, 1: accepted
const std::vector< int >& Sampler::getAcceptedHistory() const {
    return m_accepted;
};

std::vector< ParameterDescription> Sampler::getStateDescription() const { return m_stateDescription; };

void Sampler::reserveMemory(uint historySize){
    m_chain.reserve(historySize);
    m_proposalHistory.reserve(historySize);
    m_acceptanceProbs.reserve(historySize);
    m_logpriorVals.reserve(historySize);
    m_loglikelihoodVals.reserve(historySize);
    m_logposteriorVals.reserve(historySize);
    m_accepted.reserve(historySize);
    m_forced.reserve(historySize);
}

void Sampler::temper(double factor){
    m_tempered = true;
    for(int ii=0; ii<m_forwardObsCovMat.size(); ++ii){
        for(int jj=0; jj<m_forwardObsCovMat.at(ii).size(); ++jj){
            m_forwardObsCovMat.at(ii).at(jj)*=factor;
        }
    }
};


MultiThreadedSampler::MultiThreadedSampler(std::vector<std::shared_ptr<Sampler>> samplersVec):
    m_latch(samplersVec.size()), n_threads(samplersVec.size()), m_samplerVec(samplersVec){
    };
void MultiThreadedSampler::run_for(size_t n_steps, bool noProgressBar){

    indicators::DynamicProgress<indicators::ProgressBar> bars{};
    bars.set_option(indicators::option::HideBarWhenComplete{false});
   
    std::vector<std::unique_ptr<indicators::ProgressBar>> uniqueSingleBars;
    for(int i=0; i<n_threads; ++i){
        auto newBar = makeProgressBar(n_steps, "Chain #" + std::to_string(i) + " ("+ m_samplerVec.at(i)->name + ")");
        uniqueSingleBars.push_back(std::move(newBar));
    }
    for(int i = 0; i<uniqueSingleBars.size(); ++i){
        bars.push_back(*(uniqueSingleBars.at(i)));
    }
    
    for(int i = 0; i<m_samplerVec.size(); ++i){
      
        auto s_ptr = m_samplerVec.at(i);
        auto job = [this, s_ptr, &bars, i](size_t n_steps, bool noProgressBar){
                auto progressTicker = [this, i, &bars](){bars[i].tick();};
                s_ptr->run_for(n_steps, noProgressBar, progressTicker);
                m_latch.count_down();
            };
      
        if(!s_ptr){spdlog::warn("sampler ptr is null");}
        m_threads.emplace_back(job, n_steps, noProgressBar);
    }
    for(auto& thr: m_threads){
        thr.join();
    }
};

std::vector<std::shared_ptr<Sampler>> MultiThreadedSampler::getSamplers(){
    // m_latch.wait();
    return m_samplerVec;
}

