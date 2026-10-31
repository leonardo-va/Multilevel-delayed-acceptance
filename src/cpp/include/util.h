#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

namespace stats {
    template <typename Container>
    auto mean(const Container& data) {
        if (data.empty()) return 0.0;
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        return sum / data.size();
    }

    template <typename Container>
    auto sample_variance(const Container& data) {
        if (data.size() < 2) return 0.0;
        
        double avg = mean(data);
        double sq_sum = std::accumulate(data.begin(), data.end(), 0.0, [avg](double acc, double x) {
            return acc + (x - avg) * (x - avg);
        });
        
        return sq_sum / (data.size() - 1);
    }
}

namespace plot{
    template<typename domainType>
    std::vector<domainType> evaluate1DFunction(const std::vector<domainType>& grid, std::function<domainType(domainType)> func){
        std::vector<domainType> result(grid.size());
        for(int i = 0; i<grid.size(); ++i){
            result.at(i) = func(grid.at(i));
        }
        return result;
    }
}


