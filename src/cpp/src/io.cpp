#include <io.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <optional>
#include <limits>
#include <sampler.h>
#include <spdlog/spdlog.h>
#include <fmt/ranges.h>
#include <filesystem>


/// @brief write a csv with states and history of a vector of samplers. Expects the lowest level sampler first, in case of multilevel sampler.
/// @param samplerVec vector of pointers to sampler
void writeSamplerOutput(std::vector<Sampler*> samplerVec, 
                        std::string outFolder, 
                        bool removeBurnin,   
                        std::string mode,
                        int ignoreLastN
                    ){
    if(removeBurnin){
        spdlog::warn("write sampler output:: removeBurnin should no longer be used, burnin is managed by the sampler");
    }
    std::filesystem::path out = outFolder;
    if(!std::filesystem::is_directory(out)){std::filesystem::create_directories(out);}
    int samplerId = 0;
    for(const auto& s_ptr: samplerVec){
        auto stateDescription = s_ptr->getStateDescription();
        std::vector<std::string> csvHeader;
        for (const auto& paramDescr: stateDescription){
            for(const auto& subIdx: paramDescr.getSubsetIndices()){
                std::string headerEntry = paramDescr.m_name_str + "_" + std::to_string(subIdx);
                csvHeader.push_back(headerEntry);
            }
        }
        csvHeader.insert(csvHeader.end(), {"probAccept", "l_prior", "l_likelihood", "l_posterior", "accepted", "forced"});
        auto rawChain = s_ptr->applyReturnFilter(s_ptr->getChain(), ignoreLastN,removeBurnin);
        auto proposalHistory = s_ptr->applyReturnFilter(s_ptr->getProposalHistory(), ignoreLastN, removeBurnin);
        auto probAccept = s_ptr->applyReturnFilter(s_ptr->getAcceptanceProbs(), ignoreLastN, removeBurnin);
        auto logpriors = s_ptr->applyReturnFilter(s_ptr->getLogPriorValues(), ignoreLastN,removeBurnin);
        auto loglikelis = s_ptr->applyReturnFilter(s_ptr->getLogLikelihoodValues(), ignoreLastN,removeBurnin);
        auto logposteriors = s_ptr->applyReturnFilter(s_ptr->getLogPosteriorValues(), ignoreLastN, removeBurnin);
        auto forcedHistory = s_ptr->applyReturnFilter(s_ptr->getForcedHistory(), ignoreLastN, removeBurnin);
        std::vector<int> acceptedHistory = s_ptr->applyReturnFilter(s_ptr->getAcceptedHistory(), ignoreLastN,removeBurnin);
        
        int outSize = rawChain.size();
        if(outSize != proposalHistory.size() || outSize != probAccept.size() || outSize != logpriors.size() || outSize != loglikelis.size() || outSize != logposteriors.size() || outSize != acceptedHistory.size()){
            spdlog::warn("Write Sampler output:: Size mismatch between output fields");
        }
        std::vector< parameter_type > csvChain;
        std::vector< parameter_type > csvProposals;
        for(int i=0; i<rawChain.size(); ++i){
            std::vector<double> line;
            for(const auto& par: rawChain[i]){
                for(const auto& sub: par){
                    line.push_back(sub);
                }
            }
            line.push_back(probAccept[i]);
            line.push_back(logpriors[i]);
            line.push_back(loglikelis[i]);
            line.push_back(logposteriors[i]);
            line.push_back(acceptedHistory[i]);
            line.push_back(forcedHistory[i]);
            csvChain.push_back(line);
        }
        for(int i=0; i<proposalHistory.size(); ++i){
            std::vector<double> line;
            for(const auto& par: proposalHistory[i]){
                for(const auto& sub: par){
                    line.push_back(sub);
                }
            }
            line.push_back(probAccept[i]);
            line.push_back(logpriors[i]);
            line.push_back(loglikelis[i]);
            line.push_back(logposteriors[i]);
            line.push_back(acceptedHistory[i]);
            line.push_back(forcedHistory[i]);
            csvProposals.push_back(line);
        }
        // if(outFolder.size() != 0){outFolder += "/";}
        std::string outPath = outFolder + s_ptr->name + "_result.csv";
        std::string outPathProposal = outFolder + s_ptr->name + "_proposals.csv";
        writeCSV(csvChain, outPath, csvHeader, mode);
        writeCSV(csvProposals, outPathProposal, csvHeader, mode);
        ++samplerId;
    }
}

void writeSamplerOutput(Sampler* samplerPtr, 
                        std::string outFolder, 
                        bool removeBurnin, 
                        std::string mode,
                        int ignoreLastN
                    ){
    std::vector<Sampler*> samplerVec{samplerPtr};
    writeSamplerOutput(samplerVec, outFolder, removeBurnin, mode, ignoreLastN);
}

void writeSamplerOutput(std::vector<std::shared_ptr<Sampler>> samplerVec, 
                        std::string outFolder, 
                        bool removeBurnin, 
                        std::string mode,
                        int ignoreLastN
                    ){
    std::vector<Sampler*> samplerVecRaw;
    for(auto& entry: samplerVec){
        samplerVecRaw.push_back(entry.get());
    }
    std::vector<Sampler*> samplerVecRawConst(samplerVecRaw.begin(), samplerVecRaw.end());
    writeSamplerOutput(samplerVecRawConst, outFolder, removeBurnin, mode, ignoreLastN);
}

void writeCSV(const std::vector<std::vector<double>>& states, const std::vector<double>& times, std::string path){
    std::ofstream solution_file(path);
    if (!solution_file.is_open()) {
        std::cout<<"Failed to save file"<<std::endl;
        exit(-1);
    }
    size_t state_size = states[0].size();
    for(int i=0; i<states.size(); ++i){
        solution_file<<times[i]<<",";
        for(int j = 0; j<state_size; ++j){
            solution_file<<states[i][j];
            if(j<state_size-1){
                solution_file<<",";
            }
        }
        solution_file<<std::endl;;
    }
}

void writeCSV(const std::vector<double>& states, const std::vector<double>& times, std::string path){
    std::vector<std::vector<double>> vectorized_states;
    for(const auto& state : states){
        vectorized_states.emplace_back(1, state);
    }
    writeCSV(vectorized_states, times, path); 
}

void writeCSV(std::vector<std::vector<double>>& vv, std::string path, std::vector<std::string> header, std::string mode){
    std::ios_base::openmode openMode;
    if(mode == "a"){
        openMode = std::ios::app;
    }
    else if(mode == "w"){
        openMode = std::ios::out;
    }
    else{
        spdlog::warn("writeCSV: unknown mode {}, use write (w) or append (a)", mode);
        return;
    }

    std::ofstream targetFile(path, openMode);
    
    if (!targetFile.is_open()) {
        spdlog::error("Failed to save file {}", path);
        return;
    }
    size_t headerSize = header.size();
    if(headerSize != 0 && mode == "w"){
        // if(headerSize < rowSize){
        //     for(int i=0; i<rowSize-headerSize; ++i){
        //         header.push_back("");
        //     }
        // }
        for(int i=0; i<headerSize; ++i){
            targetFile<<header[i];
            if(i!=headerSize-1){targetFile<<",";}
        }
        targetFile<<std::endl;;
    }
    if(vv.size() == 0){
        spdlog::warn("write csv:: passed vector is empty");
        return;
    }
    size_t rowSize = vv[0].size();
    for(int i=0; i<vv.size(); ++i){
        for(int j = 0; j<rowSize; ++j){
            targetFile<<vv[i][j];
            if(j<rowSize-1){
                targetFile<<",";
            }
        }
        targetFile<<std::endl;;
    }
}

std::vector<std::vector<double>> readCSV(const std::string& path, bool ignore_first_row, bool ignore_first_col){
  std::vector<std::vector<double>> data;
  std::ifstream file(path);
  if (!file.is_open()) {
      std::cerr << "Failed to open CSV file " << path <<"\n";
      return data;
  }

  std::string line;
  bool first_row = true;

  while (std::getline(file, line)) {
      if(ignore_first_row && first_row){ first_row = false; continue; }
      first_row = false;

      std::stringstream ss(line);
      std::string cell;
      std::vector<double> row;
      bool first_col = true;

      while (std::getline(ss, cell, ',')) {
          if(ignore_first_col && first_col){ first_col = false; continue; }
          first_col = false;
          if(cell.size() == 0){
              row.push_back(std::numeric_limits<double>::quiet_NaN());
              continue;
          }
          try{
              row.push_back(std::stod(cell));
          }catch(...){
              // non-numeric cell -> store NaN
              row.push_back(std::numeric_limits<double>::quiet_NaN());
          }
      }

      data.push_back(row); // add the row to the main vector
  }

  file.close();
  return data;
}

std::vector<std::vector<double>> readCSV(const std::filesystem::path& path, bool ignore_first_row, bool ignore_first_col){
    return readCSV(path.string(), ignore_first_row, ignore_first_col);
}


void ParameterReader::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    json root;
    f >> root;

    if (!root.is_array())
        throw std::runtime_error("Expected a JSON array at the root");

    m_params.clear();
    for (const auto& obj : root)
        m_params.push_back(parse(obj));
}

const std::vector<Parameter>& ParameterReader::getAll() const {
    return m_params;
}

std::vector<Parameter> ParameterReader::getInferred() const {
    std::vector<Parameter> result;
    for (const auto& p : m_params)
        if (p.inferred)
            result.push_back(p);
    return result;
}

Parameter ParameterReader::parse(const json& obj) {
    Parameter p;

    p.name      = obj.at("name").get<std::string>();
    p.inferred  = obj.at("inferred").get<bool>();
    p.fct_class = obj.at("temp_dependence_type").at("fct_class").get<std::string>();

    for (const auto& v : obj.at("groundtruth"))
        p.groundtruth.push_back(v.get<double>());

    // --- chain_init (optional) ---
    if (obj.contains("chain_init") && !obj.at("chain_init").is_null()) {
        std::vector<double> vals;
        for (const auto& v : obj.at("chain_init"))
            vals.push_back(v.get<double>());
        p.chainInit = std::move(vals);
    }

    // --- proposal_noise (optional) ---
    if (obj.contains("proposal_noise") && !obj.at("proposal_noise").is_null()) {
        std::vector<double> vals;
        for (const auto& v : obj.at("proposal_noise"))
            vals.push_back(v.get<double>());
        p.proposalNoise = std::move(vals);
    }

    // --- prior (optional) ---
    if (obj.contains("prior") && !obj.at("prior").is_null()) {
        const auto& pr = obj.at("prior");
        Prior prior;
        prior.distribution = pr.at("distribution").get<std::string>();
        for (const auto& [key, arr] : pr.at("parameters").items()) {
            std::vector<double> vals;
            for (const auto& v : arr)
                vals.push_back(v.get<double>());
            prior.parameters[key] = std::move(vals);
        }
        p.prior = std::move(prior);
    }

    return p;
}

void printPrior(const Prior& prior)
{
    spdlog::info("Prior {{");
    spdlog::info("  distribution : {}", prior.distribution);
    spdlog::info("  parameters   :");

    for (const auto& [name, values] : prior.parameters) {
        spdlog::info("    {} = [{}]", name, fmt::join(values, ", "));
    }

    spdlog::info("}}");
}


void printParameter(const Parameter& param)
{
    spdlog::info("Parameter {{");
    spdlog::info("  name          : {}", param.name);
    spdlog::info("  fct_class     : {}", param.fct_class);
    spdlog::info("  inferred      : {}", param.inferred);
    spdlog::info("  groundtruth   : [{}]", fmt::join(param.groundtruth, ", "));

    if (param.prior) {
        spdlog::info("  prior         :");
        printPrior(*param.prior);
    } else {
        spdlog::info("  prior         : none");
    }

    if (param.chainInit) {
        spdlog::info("  chainInit     : [{}]", fmt::join(*param.chainInit, ", "));
    } else {
        spdlog::info("  chainInit     : none");
    }

    if (param.proposalNoise) {
        spdlog::info("  proposalNoise : [{}]", fmt::join(*param.proposalNoise, ", "));
    } else {
        spdlog::info("  proposalNoise : none");
    }

    spdlog::info("}}");
}