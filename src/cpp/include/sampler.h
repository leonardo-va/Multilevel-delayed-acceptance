#pragma once

#include <vector>
#include <odeModel.h>
#include <numbers>
#include <proposalMechanisms.h>
#include <random>
#include <types.h>
#include <distributions.h>
#include <thread>
#include <latch>
#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>


struct SamplerTiming{
    std::chrono::microseconds historyUpdate{0};
    std::chrono::microseconds acceptanceProbability{0};
    std::chrono::microseconds forward{0};
    std::chrono::microseconds logPrior{0};
    std::chrono::microseconds logLikelihood{0};
    std::chrono::microseconds proposal{0};
};

class Sampler{
    public:

    void forward(const std::vector< parameter_type >&);
    double log_likelihood(const std::vector< parameter_type >&);
    double log_prior_eval(const std::vector< parameter_type >&);
    double log_prior_eval( parameter_type );
    double log_posterior_eval(const std::vector<parameter_type>&);
    double acceptanceProb(const std::vector< parameter_type >&);
    void step();
    void dumpHistory();
    void run_for(size_t n_steps, bool noProgressBar = false, std::function<void()> progressTicker = nullptr);

    void registerModel(SEIRModel model);
    void add_inferred_param(ParameterDescription paramDescr, const multiNormalIndependentDist& priorDens);
    void add_inferred_param(ParameterDescription& paramDescr);
    void addData(const std::vector< state_type >& data);
    void add_observed_quantity(const std::vector<double> qoi_lc, double stddev);
    void set_solverSettings(const solverSettings& settings);
    void set_initialState(const std::vector< parameter_type >& initialState);
    void set_nextState(const std::vector< parameter_type >& newState);
    void set_proposalMechanism(ProposalMechanismBase* propose);
    void set_prior(std::shared_ptr<distribution> priorDist);
    void set_burnin(uint burnin, bool removeFromHistory = true);
    void set_maximumChainSize(uint maxSize){ m_maximumChainSize = maxSize; };
    void set_outputFolder(std::string outFolderPath);
    void set_minimumHistorySize(uint minSize);
    // avoids posterior calculation if the proposed state is within this absolute tolerance, default is 10^(-12)
    void set_newStateTolerance(double atol){m_newStateTolerance=atol;}

    void temper(double factor);
    
    void reserveMemory(uint historySize);

    const std::vector<parameter_type>& getCurrentState() const;
    double getCache(std::string what) const;
    const std::vector< std::vector< parameter_type > >& getChain() const;
    const std::vector< std::vector< parameter_type > >& getProposalHistory() const;
    const std::vector< double >& getAcceptanceProbs() const;
    const std::vector< double >& getLogPriorValues() const;
    const std::vector< double >& getLogLikelihoodValues() const;
    const std::vector< double >& getLogPosteriorValues() const;
    const std::vector< int >& getAcceptedHistory() const;
    const std::vector< int >& getForcedHistory() const { return m_forced; };
    const SEIRObserver& getForwardObserver() const {return m_forwardObserver.value();}
    std::shared_ptr<distribution> getPrior() const {return m_prior;}
    const RingBuffer<SamplerTiming, 100>& getTimings() const {return m_timings;};

    std::vector< ParameterDescription> getStateDescription() const;

    template<typename ContainerType>
    ContainerType applyReturnFilter(ContainerType container, int removeLastN = 0, bool removeBurnin = false) const{
        std::vector< bool > selectIndices(m_chain.size(), true);
        auto fullAcceptHistory = getAcceptedHistory();

        if(removeBurnin){
            int burninRemovedCounter = 0;
            for(int i = 0; i<m_chain.size(); ++i){
                if(selectIndices.at(i) == false){continue;}
                if(burninRemovedCounter < m_burnin){
                    selectIndices.at(i) = false;
                    ++burninRemovedCounter;
                }
            }
        }

        int lastNRemovedCounter = 0;
        for(int i = m_chain.size()-1; i>=0 && lastNRemovedCounter < removeLastN; --i){
            selectIndices.at(i) = false;
            ++lastNRemovedCounter;
        }

        int outSize = std::count(selectIndices.begin(), selectIndices.end(), true);
        if(outSize == container.size()){
            return container;
        }
        
        ContainerType outContainer(outSize);
        auto outIdx = 0;
        for(int i = 0; i<container.size(); ++i){
            if(selectIndices.at(i)){
                outContainer.at(outIdx) = container.at(i);
                ++outIdx;
            }
        }
        return outContainer;
    };

    std::string name = "default_name";
    int level = 0;

    private:
    // chain history and cached data
    std::vector< std::vector< parameter_type > > m_chain;
    std::vector< std::vector< parameter_type > > m_proposalHistory;
    std::vector< double > m_acceptanceProbs;
    std::vector< double > m_logpriorVals;
    std::vector< double > m_loglikelihoodVals;
    std::vector< double > m_logposteriorVals;
    std::vector< int > m_accepted;
    std::vector< int > m_forced;
    std::string m_outFolderPath;
    RingBuffer<SamplerTiming, 100> m_timings;
   
    double m_cachedLogPosterior;
    double m_cachedLogPrior;
    double m_cachedLogLikelihood;
    double m_currentStateLogPosterior;
    double m_currentStateLogPrior;
    double m_currentStateLogLikelihood;
    double m_newStateTolerance = 1e-12;
    // observer(also chached)
    std::optional<SEIRObserver> m_forwardObserver;
    std::vector<double> m_forwardObsTimes;
    std::vector<state_type> m_forwardObsStates;
    void clearHistory();
    void eraseHistory(uint keepLastN);
    
    // set on setup time
    
    // burnin
    uint m_burnin = 0;
    uint m_stepCounter = 0;
    uint m_keepOnErase = 100;
    bool m_removeBurninFromHistory = true;
    uint m_maximumChainSize = std::numeric_limits<uint>::max();
    
    // model
    SEIRModel m_model;
    
    // proposal
    ProposalMechanismBase* m_proposalMechanism{nullptr};
    
    // prior
    std::shared_ptr<distribution> m_prior;
    std::vector< ParameterDescription > m_stateDescription;
    std::vector< ParameterDescription > m_fineDescription;
    std::vector< ParameterDescription > m_coarseDescription;
    
    // data
    std::vector< state_type > m_data;
    std::vector<double> m_measure_times;
    std::vector<double> m_forwardObsSigmaSquared;
    std::vector<std::vector<double>> m_forwardObsCovMat;
    int m_numberQois{0};
    bool m_tempered = false;
    
    // solver
    std::optional<solverSettings> m_solverSettings;
    
    // set on object construction
    boost::numeric::odeint::runge_kutta4 < state_type > m_stepper;
    std::mt19937 m_rngEngine{std::random_device{}()};
    std::uniform_real_distribution<double> m_uniform01{0.0, 1.0};
};

class DynamicProgressBars;
class MultiThreadedSampler{
public:
    MultiThreadedSampler(std::vector<std::shared_ptr<Sampler>> samplersVec);
    void run_for(size_t n_steps, bool noProgressBar=false);
    std::vector<std::shared_ptr<Sampler>> getSamplers();

private:
    std::vector<std::shared_ptr<Sampler>> m_samplerVec;
    std::vector<std::thread> m_threads;
    int n_threads;
    std::latch m_latch;
    indicators::DynamicProgress<indicators::ProgressBar> m_bars;
};


