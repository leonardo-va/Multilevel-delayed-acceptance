#pragma once

#include <types.h>
#include <odeModel.h>
#include <random>
#include <distributions.h>

// forward declaration, incomplete type
class Sampler;

class ProposalMechanismBase{
    public:
    virtual std::vector<parameter_type> operator() (const std::vector<parameter_type>& currentState) = 0;
    virtual double acceptanceProbability(Sampler* sampler, const std::vector<parameter_type>& proposedState) = 0;
    virtual const Sampler* getCoarseSampler() const {return nullptr;};
    virtual void setOwningSampler(Sampler& owningSampler);
    
    Sampler* m_owningSampler;
    protected:
};

class RandomWalkProposal : public ProposalMechanismBase{
    public:
    RandomWalkProposal(std::vector<std::vector<double>> vv_stddev): m_vvstddev(vv_stddev){};
    std::vector<parameter_type> operator()(const std::vector<parameter_type>& currentState) override;
    double acceptanceProbability(Sampler* sampler, const std::vector<parameter_type>& proposedState) override;
    
    private:
        std::vector<std::vector<double>> m_vvstddev;
};

class PcnProposal : public ProposalMechanismBase{
    public:
    PcnProposal();
    PcnProposal(double beta, std::shared_ptr<multiNormalDist> priorDist);
    std::vector<parameter_type> operator() (const std::vector<parameter_type>& currentState) override;
    double acceptanceProbability(Sampler* sampler, const std::vector<parameter_type>& proposedState) override;
    void setOwningSampler(Sampler& owningSampler);
    

    private:
    double m_beta;
    double m_meanFactor;
    std::shared_ptr<multiNormalDist> m_priorDist;
};

class FineLevelProposal : public ProposalMechanismBase{
    
    public:
    FineLevelProposal(Sampler* coarseSampler, 
        size_t subchainLength, 
        std::vector<ParameterDescription> fullStateDescription, 
        ProposalMechanismBase* finePartMechanism);
    
    std::vector< parameter_type > extractCoarseState( const std::vector< parameter_type >& fineState ) const;
    void buildFineState(std::vector< parameter_type >& fineState, const std::vector< parameter_type >& coarseState ) const;
    std::pair< std::vector< parameter_type >, std::vector< parameter_type > > splitState(const std::vector< parameter_type >& fullState) const;
    std::vector< parameter_type > combineToFullState(const std::vector< parameter_type >& fineState, const std::vector< parameter_type >& coarseState) const;
    std::vector<parameter_type> operator()(const std::vector<parameter_type>& currentState) override;
    double acceptanceProbability(Sampler* sampler, const std::vector<parameter_type>& proposedState) override;
    const Sampler* getCoarseSampler() const override;
    void setOwningSampler(Sampler& owningSampler) override;
    
    
    private:
    Sampler* m_coarseSampler;
    size_t m_subchainLength;
    std::vector<ParameterDescription> m_coarseStateDescription;
    std::vector<ParameterDescription> m_fullStateDescription;
    std::vector<ParameterDescription> m_fineStateDescription;
    ProposalMechanismBase* m_finePartMechanism;
    std::vector<int> m_fullStateIndexCorrespondence;
    enum class tags{FINE, SHARED, COARSE};
    std::vector< tags > m_nameTags;
    std::vector< std::vector< tags > > m_mixedTags;
    std::mt19937 m_rngEngine{std::random_device{}()};
    std::uniform_int_distribution<> m_randomSubchainIndex;
    int m_currentSelectedSubchainIndex{0};
};
