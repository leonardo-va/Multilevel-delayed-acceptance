#pragma once
#include <vector>
#include <string>
#include <optional>
#include <map>

typedef std::vector<double> parameter_type;


// ---------------------------------------------------------------------------
// Prior
// ---------------------------------------------------------------------------
struct Prior {
    std::string distribution;                       // e.g. "Normal"
    std::map<std::string, std::vector<double>> parameters;  // e.g. {"mu": [...], "sigma": [...]}
};

// ---------------------------------------------------------------------------
// Temperature dependent Parameter
// ---------------------------------------------------------------------------
struct Parameter {
    std::string              name;
    std::string              fct_class;             // e.g. "Briere", "Quadratic", "Constant"
    bool                     inferred;
    std::vector<double>      groundtruth;
    std::optional<Prior>     prior;                  // only present when inferred == true
    std::optional<std::vector<double>> chainInit;
    std::optional<std::vector<double>> proposalNoise;            
};

struct QuantityOfInterest{
    std::string name{""};
    std::vector<double> linearCombination;
    double observationNoise{1.0};
};

// ringbuffer

template <typename T, std::size_t N>
class RingBuffer {
    std::array<T, N> data;
    std::size_t next = 0;
    
    public:
    void push(T value) {
        data[next] = std::move(value);
        next = (next + 1) % N;
    }
    T& latest() {
        return data[(next + N - 1) % N];
    }
    const std::array<T,N>& getData() const {return data;}
};