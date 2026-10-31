#include <odeModel.h>
#include <sampler.h>
#include <types.h>
#include <io.h>
#include <spdlog/spdlog.h>
#include <filesystem>

// example code for MLDA with two levels

int main(){
    int numberOfChains = 8;
    int numberOfSamples = 100;
    int burnin = 10;
    int subchainLength = 8;

    std::filesystem::path outputFolder = "./out";
    std::filesystem::path pathToData = "dataJM.csv";
    
    // default model parameters are used here (groundtruth)
    auto model_coarse = std::make_shared<SEIRModel>();
    model_coarse->set_evaluator(std::make_shared<FullModelEvaluator>());

    auto model_fine = std::make_shared<SEIRModel>(); 
    model_fine->set_evaluator(std::make_shared<FullModelEvaluator>());

    // define the settings for the ODE-solver on each level, in the format {startTime, endTime, stepSize, initialValue}
    state_type initialValue{320000,2000,11000,0,1000,5000,0,0,0};
    solverSettings settings_coarse  (0., 200., 0.01,  initialValue);
    solverSettings settings_fine    (0., 720., 0.005, initialValue);

    
    // describe the noise model for the observations. Observations are considered linear combinations 
    // of a state of the ode. Here, the operators J and M are defined, with the Gaussian noise standard deviation, 
    // as in section "data generation"
    std::vector<QuantityOfInterest> observedQuantities_coarse = { 
        {"juveniles",               {0,1,0,0,0,0,0,0,0},    4000.    },
        {"total mosquitoes",        {0,0,1,1,1,0,0,0,0},    30000.   }
    };
    std::vector<QuantityOfInterest> observedQuantities_fine = {
        {"juveniles",               {0,1,0,0,0,0,0,0,0},    4000.    },
        {"total mosquitoes",        {0,0,1,1,1,0,0,0,0},    30000.   }
    };

    // Define your data. The rows of this matrix are the individual measurements. The first column is the time of
    // the measurement. Then the measured quantities follow in the order described above.
    std::vector<state_type> data = readCSV(pathToData);
    // ....
    // list the inferred parameters for each level. The coarse part must be a subset of the full parameter set 
    // on the fine level.
    std::vector<ParameterDescription> inferredParameters_coarse{
        ParameterDescription{   SEIRModel::ParamName::a     },
        ParameterDescription{   SEIRModel::ParamName::lf_M  }
    };
    std::vector<ParameterDescription> inferredParameters_fine{
        ParameterDescription{   SEIRModel::ParamName::a     },
        ParameterDescription{   SEIRModel::ParamName::lf_M  }
    };
    
    // Construct a gaussian prior. The dimensions of the parameter space are ordered as in the
    // parameter description above. "a" is 3-dimensional, "lf_M" is 2-dimensional. So, the
    // mean would be in the format: [a_0, a_1, a_2, lf_M_0, lf_M_1], and accordingly the covariance. 
    // In this example, the parameter space on both levels is the same. A single prior can be defined
    // here for both levels. If the spaces are different i.e. theta = (theta_fine, theta_coarse), 
    // a prior for all three sets of unknowns theta, theta_fine, and theta_coarse must be defined.
    parameter_type priorMean{2., 2., 40., 3., 35.};
    std::vector<parameter_type> priorCov{
        {1,0,0,0,0},
        {0,4,0,0,0},
        {0,0,5,0,0},
        {0,0,0,2,0},
        {0,0,0,0,2}
    };
    auto prior = std::make_shared<multiNormalDist>(priorMean, priorCov);

    // Define the proposal strategies. Here, pCN proposals are used, which take as arguments the step size
    // parameter beta, and the prior as Gaussian reference measure. Each sampler needs one individually,
    // and the objects need to be kept alive here because ownership is not transferred. The fine samplers
    // need a strategy both for the full parameter space theta, and for the fine part theta_fine.
    // If the parameter spaces on the coarse and fine levels are the same, the fine part propsal can be 
    // default initialized, it is then empty and handled accordingly.

    // coarse part proposals
    double pcnBeta_coarse{0.2}, pcnBeta_fine{0.2};
    std::vector<std::shared_ptr<ProposalMechanismBase>> coarseProposals;
    for(int i = 0; i<numberOfChains; ++i){
        coarseProposals.emplace_back(std::make_shared<PcnProposal>(pcnBeta_coarse, prior));
    }

    // fine part proposals are empty, since the theta_fine is empty in this example
    std::vector<std::shared_ptr<ProposalMechanismBase>> fineProposals;
    for(int i = 0; i<numberOfChains; ++i){
        fineProposals.emplace_back(std::make_shared<PcnProposal>());
    }
    
    // Set up the coarse samplers. One sampler for each of the chains.
    std::vector<std::shared_ptr<Sampler>> coarseSamplers;
    for(int i = 0; i<numberOfChains; ++i){
        auto sampler = std::make_shared<Sampler>();
        sampler->name = "coarse_" + std::to_string(i);
        sampler->set_outputFolder(outputFolder / "coarse" / "");
        
        sampler->registerModel(*model_coarse);
        sampler->set_solverSettings(settings_coarse);
        
        sampler->addData(data);
        for(auto& quantity: observedQuantities_coarse){
            sampler->add_observed_quantity(quantity.linearCombination, quantity.observationNoise);
        }
        
        for(auto& inferred: inferredParameters_coarse){
            sampler->add_inferred_param(inferred);
        }
        sampler->set_prior(prior);
        
        sampler->set_proposalMechanism(coarseProposals.at(i).get());
        
        sampler->set_burnin(burnin);
        sampler->reserveMemory(numberOfSamples*subchainLength);
        
        coarseSamplers.push_back(sampler);
    }

    // Now, with the coarse and fine parts, the multilevel proposal for theta = (theta_fine, theta_coarse) can be defined.
    // The multilevel proposal needs access to the coarse sampler, thus they had to be defined previously.
    // The other arguments are the subchain length, the description of the parameter space of the current level,
    // and the fine part proposal.
    // More than two levels can be defined, by simply setting up another multilevel proposal and passing the sampler
    // on the previous level as coarse sampler.
    
    std::vector<std::shared_ptr<ProposalMechanismBase>> multiLevelProposals;
    for( int i = 0; i<numberOfChains; ++i){
        auto mlProposal = std::make_shared<FineLevelProposal>(coarseSamplers.at(i).get(), 
                                                            subchainLength, 
                                                            inferredParameters_fine, 
                                                            fineProposals.at(i).get());
        multiLevelProposals.push_back(mlProposal);
    }

    // Define the fine samplers:
    std::vector<std::shared_ptr<Sampler>> fineSamplers;
    for(int i = 0; i<numberOfChains; ++i){
        auto sampler = std::make_shared<Sampler>();
        sampler->name = "fine_" + std::to_string(i);
        sampler->set_outputFolder(outputFolder / "fine" / "");
        
        sampler->registerModel(*model_fine);
        sampler->set_solverSettings(settings_fine);
        
        sampler->addData(data);
        for(auto& quantity: observedQuantities_fine){
            sampler->add_observed_quantity(quantity.linearCombination, quantity.observationNoise);
        }
        
        for(auto& inferred: inferredParameters_fine){
            sampler->add_inferred_param(inferred);
        }
        sampler->set_prior(prior);
        
        sampler->set_proposalMechanism(multiLevelProposals.at(i).get());
        
        sampler->set_burnin(burnin);
        sampler->reserveMemory(numberOfSamples);
        fineSamplers.push_back(sampler);
    }

    // Initialize all samplers by setting an initial state for each inferred parameter,
    // in the order as they were added to the sampler
    std::vector<state_type> initialState_coarse{
        {2,2,40},
        {3,35}
    };
    std::vector<state_type> initialState_fine{
        {2,2,40},
        {3,35}
    };
    for(int i = 0; i<numberOfChains; ++i){
        coarseSamplers.at(i)->set_initialState(initialState_coarse);
        fineSamplers.at(i)->set_initialState(initialState_fine);
    }

    // To run the chains in parallel, define the workers:
    auto mtCoarse = MultiThreadedSampler(coarseSamplers);
    auto mtFine   = MultiThreadedSampler(fineSamplers);

    // Each level needs to go through its burnin, before the next level can run. If there was no burnin period
    // defined previously, this can be skipped.
    mtCoarse.run_for(burnin+1);
    // For the finest level, burnin can be combined with sampling
    mtFine.run_for(numberOfSamples + burnin);

    // Write the output in a csv
    writeSamplerOutput(mtFine.getSamplers(), outputFolder / "fine" / "", false, "a", 0);

    return 0;
}