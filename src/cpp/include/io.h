#pragma once

#include <types.h>
#include <vector>
#include <string>
#include <memory>
// parameterreader
#include <map>
#include <optional>
#include <stdexcept>
#include <fstream>
#include <nlohmann/json.hpp>
#include <filesystem>
using json = nlohmann::json;

class Sampler;

void writeSamplerOutput(std::vector<Sampler*> samplerVec, 
                            std::string outFolder = "", 
                            bool removeBurnin = false, 
                            std::string mode = "w",
                            int ignoreLastN = 0
                        );
void writeSamplerOutput(Sampler* samplerPtr, std::string outFolder = "", 
                            bool removeBurnin = false, 
                            std::string mode = "w",
                            int ignoreLastN = 0
                        );
void writeSamplerOutput(std::vector<std::shared_ptr<Sampler>> samplerVec, std::string outFolder, 
                            bool removeBurnin, 
                            std::string mode = "w",
                            int ignoreLastN = 0
                        );
void writeCSV(const std::vector<std::vector<double>>& states, const std::vector<double>& times, std::string path);
void writeCSV(const std::vector<double>& states, const std::vector<double>& times, std::string path);
void writeCSV(std::vector<std::vector<double>>& vv, std::string path, std::vector<std::string> header = std::vector<std::string>{}, std::string mode = "w");
// Read CSV into rows of doubles. The inner vectors are rows.
// If the file has header labels in first row/column set the corresponding flag to true to ignore them.
std::vector<std::vector<double>> readCSV(const std::string& path, bool ignore_first_row = false, bool ignore_first_col = false);
std::vector<std::vector<double>> readCSV(const std::filesystem::path& path, bool ignore_first_row, bool ignore_first_col);

template <typename rb>
void writeTimings(const rb& timings, const std::string& filename)
{
    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    file << "historyUpdate,"
         << "acceptanceProbability,"
         << "forward,"
         << "logPrior,"
         << "logLikelihood,"
         << "proposal\n";

    for (std::size_t i = 0; i < timings.getData().size(); ++i) {
        const auto& timing = timings.getData()[i];

        file << timing.historyUpdate.count() << ','
             << timing.acceptanceProbability.count() << ','
             << timing.forward.count() << ','
             << timing.logPrior.count() << ','
             << timing.logLikelihood.count() << ','
             << timing.proposal.count() << '\n';
    }
}

// ---------------------------------------------------------------------------
// ParameterReader
// ---------------------------------------------------------------------------
class ParameterReader {
public:
    void load(const std::string& path);

    // Returns all parameters
    const std::vector<Parameter>& getAll() const;

    // Returns only parameters tagged "inferred": true
    std::vector<Parameter> getInferred() const;

private:
    std::vector<Parameter> m_params;
    static Parameter parse(const json& obj);
};

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------
void printPrior(const Prior& prior);
void printParameter(const Parameter& param);
