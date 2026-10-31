#include <proposalMechanisms.h>
#include <stdexcept>
#include <sampler.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <magic_enum/magic_enum.hpp>
// preconditioned crank nikolson
// tempering on coarse level


void ProposalMechanismBase::setOwningSampler(Sampler& owningSampler){
        m_owningSampler = &owningSampler;
}

std::vector<parameter_type> RandomWalkProposal::operator()(const std::vector<parameter_type>& currentState){
    int proposalSize = m_vvstddev.size();
    if(currentState.size() != proposalSize){
        spdlog::error("proposalRW(sampler {})", m_owningSampler->name);
        throw std::invalid_argument("proposalRW:: mc chain state size doesnt match noise vector size");
    }
    for(int i=0; i<currentState.size(); ++i){
        if(currentState[i].size() != m_vvstddev[i].size()){
            throw std::invalid_argument("proposalRW:: mc chain state size doesnt match noise vector size");
        }
    }
    std::vector< parameter_type > newProposal(proposalSize);
    for(int j=0; j<proposalSize; ++j){
        int par_size = m_vvstddev[j].size();
        // std::vector<double> v_var(par_size, 0.0);
        std::vector<std::vector<double>> covMat;

        for(int i=0; i<par_size; ++i){
            // v_var[i] = m_vvstddev[j][i]*m_vvstddev[j][i];
            std::vector<double> covRow(par_size, 0.0);
            covRow.at(i) = m_vvstddev[j][i] * m_vvstddev[j][i];
            covMat.push_back(covRow);
        }
        auto multNormal = multiNormalDist(currentState[j], covMat);
        // auto multNormal = multiNormalIndependentDist(currentState[j], v_var);
        newProposal[j] = multNormal.getSample();
    }
    return newProposal;
}

double RandomWalkProposal::acceptanceProbability(Sampler* sampler, const std::vector<parameter_type>& proposedState){
    double proposedLogPosterior = sampler->log_posterior_eval(proposedState);
    double currentLogPosterior = sampler->getLogPosteriorValues().back(); 
    double logAcceptance = std::min(0., proposedLogPosterior-currentLogPosterior);
    return std::exp(logAcceptance);
}

PcnProposal::PcnProposal(): m_beta(0), m_meanFactor(1), m_priorDist(std::make_shared<multiNormalDist>()){};
PcnProposal::PcnProposal(double beta, std::shared_ptr<multiNormalDist> priorDist){
    m_beta = beta;
    m_meanFactor = std::sqrt(1-m_beta*m_beta);
    m_priorDist = priorDist;
}

void PcnProposal::setOwningSampler(Sampler& owningSampler){
        if(!std::dynamic_pointer_cast<multiNormalDist>(owningSampler.getPrior())){
            spdlog::error("PcnProposal:: Pcn proposal must be used with a gaussian prior");
            return;
        }
        m_owningSampler = &owningSampler;
}

std::vector<parameter_type> PcnProposal::operator() (const std::vector<parameter_type>& currentState){
    // this is not tested for a fine/coarse split state. 
    int paramSpaceDim = m_priorDist->getDim();
    parameter_type scaledCorrectedFlattenedState(paramSpaceDim, 0.0);
    parameter_type priorMean = m_priorDist->getMean();
    parameter_type priorSample = m_priorDist->getSample();
    int flattenedIdx = 0;
    for(int i = 0; i < currentState.size(); ++i){
        for(int j = 0; j < currentState.at(i).size(); ++j){
            scaledCorrectedFlattenedState.at(flattenedIdx) = m_meanFactor * (currentState.at(i).at(j) - priorMean.at(flattenedIdx));
            ++flattenedIdx;
        }
    }
    if(flattenedIdx != paramSpaceDim ){
        spdlog::warn("PcnProposal:: Proposal Index Error during flattening the current state. Flattened index: {}, param space dim: {}",
                    flattenedIdx, paramSpaceDim);
    }
    parameter_type flattenedProposal(paramSpaceDim, 0.0);
    for(int i = 0; i < paramSpaceDim; ++i){
        flattenedProposal.at(i) = priorMean.at(i) + scaledCorrectedFlattenedState.at(i) + (priorSample.at(i) - priorMean.at(i))*m_beta;
        //im not sure about this other papers say sqrt(1-b²)theta+b*N(0,priorCov).I implemented priorMu + sqrt(1-b²)(theta-priorMu)+b*(N(priorMu, priorCov)-priorMu)
    }
    flattenedIdx = 0;
    std::vector<parameter_type> foldedProposal(currentState.size());
    for(int i = 0; i < currentState.size(); ++i){
        int parSize = currentState.at(i).size();
        parameter_type paramEntry(parSize);
        for(int j = 0; j < parSize; ++j){
            paramEntry.at(j) = flattenedProposal.at(flattenedIdx);
            ++ flattenedIdx;
        }
        foldedProposal.at(i) = paramEntry;
    }
    if(flattenedIdx != flattenedProposal.size()){
        spdlog::error("PcnProposal:: Proposal Index Error during folding the proposal");
    }
    for(int i = 0; i < foldedProposal.size(); ++i){
        if(foldedProposal.at(i).size() != currentState.at(i).size()){
            spdlog::error("PcnProposal:: Proposal Index Error, current state and proposal mismatch");
        }
    }
    return foldedProposal;
}

double PcnProposal::acceptanceProbability(Sampler* sampler, const std::vector<parameter_type>& proposedState){
    double currentLL = sampler->getLogLikelihoodValues().back();
    sampler->log_posterior_eval(proposedState);
    // The proposed LL is stored in cache after log_posterior_eval. This is to avoid calculating it again. There might be
    // some optimization potential by not calculating the posterior, but only the likelihood
    double proposedLL = sampler->getCache("logLikelihood");
    double logAcceptance = std::min(0., proposedLL-currentLL);
    return std::exp(logAcceptance);
}


FineLevelProposal::FineLevelProposal(Sampler* coarseSampler, size_t subchainLength, std::vector<ParameterDescription> fullStateDescription, ProposalMechanismBase* finePartMechanism): m_coarseSampler(coarseSampler), 
                                                                                                                                    m_subchainLength(subchainLength),
                                                                                                                                    m_fullStateDescription(fullStateDescription),
                                                                                                                                    m_finePartMechanism(finePartMechanism){
    m_coarseStateDescription = m_coarseSampler->getStateDescription();
    // find the position of the coarse level state in the full state. Throw if it is not a subset
    std::vector<int> fullStateIndexCorrespondence(m_coarseStateDescription.size(), -1);
    for(int i=0; i<m_coarseStateDescription.size(); ++i){
        const ParameterDescription entry_coarse = m_coarseStateDescription[i];
        bool foundFineCorrespondence = false;
        for(int j = 0; j < m_fullStateDescription.size(); ++j){
            const ParameterDescription entry_fine = m_fullStateDescription[j];
            if(entry_fine.m_name == entry_coarse.m_name){
                if(entry_fine.size() < entry_coarse.size()){
                    throw std::runtime_error("FineLevelProposal:: Coarse level parameters must be a subset of fine level parameters");
                }
                if(entry_fine.size() != entry_fine.m_totalSize){
                    for(const auto& index: entry_coarse.getSubsetIndices()){
                        auto fineSSind = entry_fine.getSubsetIndices();
                        auto indexFineCorrespondence = std::find(fineSSind.begin(), fineSSind.end(), index);
                        for(const auto&val: fineSSind){std::cout<<val<<"!!"<<std::endl;;}
                        if(!fineSSind.contains(index)){
                            throw std::runtime_error("2FineLevelProposal:: Coarse level parameters must be a subset of fine level parameters");
                        }
                    }
                }
                foundFineCorrespondence = true;
                fullStateIndexCorrespondence[i] = j;
            }
        }
        if(!foundFineCorrespondence){
            throw std::runtime_error("FineLevelProposal:: Coarse level parameters must be a subset of fine level parameters");
        }
    }
    m_fullStateIndexCorrespondence = fullStateIndexCorrespondence;
    // make fine state description
    for(int i=0; i<m_fullStateDescription.size(); ++i){  
        bool foundCoarseEntry = false;
        for(const auto& coarseEntry: m_coarseStateDescription){
            if(m_fullStateDescription[i].m_name == coarseEntry.m_name){
                auto fullIndexSet = m_fullStateDescription[i].getSubsetIndices();
                auto coarseIndexSet = coarseEntry.getSubsetIndices();
                std::set<int> fineIndexSet;
                for(const auto& index: fullIndexSet){
                    if(!coarseIndexSet.contains(index)){
                        fineIndexSet.insert(index);
                    }
                }
                ParameterDescription fineEntry = ParameterDescription{coarseEntry.m_name, fineIndexSet};
                m_fineStateDescription.push_back(fineEntry);
                foundCoarseEntry = true;
                break;
            }
        }
        if(!foundCoarseEntry){
            m_fineStateDescription.push_back(m_fullStateDescription[i]);
        }
    } 
    
    // for seperating coarse/fine parts
    std::vector< tags > nameTags;
    std::vector< std::vector< tags > > mixedSubsetTags; 
    for(int i = 0; i<m_fullStateDescription.size(); ++i){
        bool foundCoarseEntry = false;
        for(const auto& coarseEntry: m_coarseStateDescription){
            if(m_fullStateDescription[i] == coarseEntry){
                nameTags.push_back(tags::COARSE);
                foundCoarseEntry = true;
                break;
            }
            if(m_fullStateDescription[i].m_name == coarseEntry.m_name){
                nameTags.push_back(tags::SHARED);
                auto fullIndexSet = m_fullStateDescription[i].getSubsetIndices();
                auto coarseIndexSet = coarseEntry.getSubsetIndices();
                std::vector< tags > mixedTags;
                for(const auto& index: fullIndexSet){
                    if(!coarseIndexSet.contains(index)){
                        mixedTags.push_back(tags::FINE);
                    }
                    else{
                        mixedTags.push_back(tags::COARSE);
                    }
                }
                foundCoarseEntry = true;
                mixedSubsetTags.push_back(mixedTags);
                break;
            }
        }
        if(!foundCoarseEntry){
            nameTags.push_back(tags::FINE);
        }
    }
    m_nameTags = nameTags;
    m_mixedTags = mixedSubsetTags;
    m_randomSubchainIndex = std::uniform_int_distribution<>(0, subchainLength-1);
};

std::vector< parameter_type > FineLevelProposal::extractCoarseState( const std::vector< parameter_type >& fineState ) const {
    std::vector< parameter_type > coarseState(m_fullStateIndexCorrespondence.size());
    for(int i=0; i<m_fullStateIndexCorrespondence.size(); ++i){
        for(const auto& subsetIdx : m_coarseStateDescription[i].m_subsetIndices){
            coarseState[i].push_back(fineState.at(m_fullStateIndexCorrespondence[i]).at(subsetIdx));
        };
    }
    return coarseState;
};

void FineLevelProposal::buildFineState(std::vector< parameter_type >& fineState, const std::vector< parameter_type >& coarseState ) const {
    for(int i=0; i<m_fullStateIndexCorrespondence.size(); ++i){
        for(int j = 0; j < m_coarseStateDescription[i].m_subsetIndices.size(); ++j){
            int subsetIdx = m_coarseStateDescription[i].m_subsetIndices[j];
            fineState.at(m_fullStateIndexCorrespondence[i]).at(subsetIdx) = coarseState.at(i).at(j);
        };
    }
};

std::pair< std::vector< parameter_type >, std::vector< parameter_type > > FineLevelProposal::splitState(const std::vector< parameter_type >& fullState) const {
    if(m_nameTags.size() != fullState.size()){
        throw std::runtime_error("FineLevelProposal:: mismatch passed state and state description");
    }
    std::vector< parameter_type > finePart;
    std::vector< parameter_type > coarsePart;
    auto mixedTags_It = m_mixedTags.begin();
    for(int i = 0; i<fullState.size(); ++i){
        if(m_nameTags.at(i) == tags::FINE){
            finePart.push_back(fullState.at(i));
        }
        if(m_nameTags.at(i) == tags::COARSE){
            coarsePart.push_back(fullState.at(i));
        }
        if(m_nameTags.at(i) == tags::SHARED){
            parameter_type coarseSubset;
            parameter_type fineSubset;
            for(int j=0; j<fullState.at(i).size(); ++j){
                if((*mixedTags_It).at(j) == tags::FINE){
                    fineSubset.push_back(fullState.at(i).at(j));
                }
                if((*mixedTags_It).at(j) == tags::COARSE){
                    coarseSubset.push_back(fullState.at(i).at(j));
                }
            }
            finePart.push_back(fineSubset);
            coarsePart.push_back(coarseSubset);
            std::advance(mixedTags_It, 1);
        }
    }
    return{finePart, coarsePart};
}

std::vector< parameter_type > FineLevelProposal::combineToFullState(const std::vector< parameter_type >& fineState, const std::vector< parameter_type >& coarseState) const {
    std::vector< parameter_type > fullState;
    auto coarseState_It = coarseState.begin();
    auto fineState_It = fineState.begin();
    auto shared_It = m_mixedTags.begin();
    for(int i=0; i<m_nameTags.size(); ++i){
        if(m_nameTags.at(i) == tags::FINE){
            fullState.push_back(*fineState_It);
            std::advance(fineState_It, 1);
        }
        if(m_nameTags.at(i) == tags::COARSE){
            fullState.push_back(*coarseState_It);
            std::advance(coarseState_It, 1);
        }
        if(m_nameTags.at(i) == tags::SHARED){
            parameter_type combined;
            auto fineIndex_It = (*fineState_It).begin();
            auto coarseIndex_It = (*coarseState_It).begin();
            for(int j = 0; j<(*shared_It).size(); ++j){
                auto tag = (*shared_It).at(j);
                if(tag == tags::FINE){
                    combined.push_back(*fineIndex_It);
                    std::advance(fineIndex_It, 1);
                }
                if(tag == tags::COARSE){
                    combined.push_back(*coarseIndex_It);
                    std::advance(coarseIndex_It, 1);
                }
            }
            fullState.push_back(combined);
            std::advance(fineState_It, 1);
            std::advance(coarseState_It, 1);
            std::advance(shared_It, 1);
        }
    }
    return fullState;
}

std::vector<parameter_type> FineLevelProposal::operator()(const std::vector<parameter_type>& currentState){
    auto[fineState, coarseState] = splitState(currentState);
    // there might be an unnecessary posterior evaluation here in set_nextState, this is measurable in the benchmark. a Two level sample with subchain length 1
    // should be about the same cost as a fine only sample + a coarse sample, but its a bit more. This posterior is probably calculated again when making the next
    // acceptance probability. With subchain length -> inf this should play less of a role.
    m_coarseSampler->set_nextState(coarseState);
    m_coarseSampler->run_for(m_subchainLength, true);
    auto coarseChain = m_coarseSampler->getChain();
    m_currentSelectedSubchainIndex = m_randomSubchainIndex(m_rngEngine);
    int coarseProposalIndex = coarseChain.size()-1-m_currentSelectedSubchainIndex;
    if(coarseProposalIndex >= coarseChain.size() || coarseProposalIndex<0){
        spdlog::error("Fine level proposal ({}):: history of coarse sampler is not large enough, make sure it went through its defined warmup", m_owningSampler->name);
    }
    std::vector< parameter_type > proposalCoarse = coarseChain.at(coarseProposalIndex);
    // std::vector< parameter_type > proposalCoarse = m_coarseSampler->getCurrentState();
    std::vector< parameter_type > proposalFine = (*m_finePartMechanism)(fineState);
    std::vector< parameter_type > proposalFull = combineToFullState(proposalFine, proposalCoarse);
    // for(auto& entry: currentState){
    //     spdlog::info("urrentState orignial {}", entry);
    // }
    // spdlog::info("proposal summary------------");
    // for(int i = 0; i < m_subchainLength; ++i){
    //     spdlog::info("{} , {}", i, coarseChain.at(coarseChain.size()-1 - i));
    // }
    // spdlog::info("chosen: i={}, {}", m_currentSelectedSubchainIndex, coarseChain.at(coarseChain.size()-1-m_currentSelectedSubchainIndex));
    return proposalFull;
}

double FineLevelProposal::acceptanceProbability(Sampler* sampler, const std::vector<parameter_type>& proposedState) {
   
    auto coarseChain = m_coarseSampler->getChain();
    auto coarseLogPosteriors = m_coarseSampler->getLogPosteriorValues();
    auto fullLogPosteriors = sampler->getLogPosteriorValues();
    double logPosteriorProposed = sampler->log_posterior_eval(proposedState);
    size_t coarseLength = coarseLogPosteriors.size();
    if(coarseLength < m_subchainLength){
        throw std::runtime_error("FineLevelProposal:: acceptance probability cant be calculated, coarse chain did not run");
    }

    double currentStateCoarseLogPost = coarseLogPosteriors.at(coarseLogPosteriors.size() - m_subchainLength -1);
    double logAccept = logPosteriorProposed - fullLogPosteriors.back() + currentStateCoarseLogPost - coarseLogPosteriors.at(coarseChain.size()-1-m_currentSelectedSubchainIndex);
    // spdlog::info("{} subchain idx {}, logPostProposed {} - logPostCurrent {}, + currentCoarseLogPost {} - coarseLogPostProposed {}", 
    //     sampler->name,
    //     m_currentSelectedSubchainIndex,
    //     logPosteriorProposed, fullLogPosteriors.back(), currentStateCoarseLogPost, coarseLogPosteriors.at(coarseChain.size()-1-m_currentSelectedSubchainIndex)
    // );
    
    return std::min(1., std::exp(logAccept));
}

const Sampler* FineLevelProposal::getCoarseSampler() const {return m_coarseSampler;};

void FineLevelProposal::setOwningSampler(Sampler& owningSampler){
    m_owningSampler = &owningSampler;
    m_finePartMechanism->m_owningSampler = &owningSampler;
};