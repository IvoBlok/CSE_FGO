#include "discreteDistribution.hpp"
#include <algorithm>
#include <numeric>

DiscreteDistribution::DiscreteDistribution(size_t size) 
    : numberOfElements(size),
      probabilities(size),
      alias(size) {

    std::vector<float> weights(size, 1.0f);
    updateDistribution(weights);
}

DiscreteDistribution::DiscreteDistribution(const std::vector<float>& weights) 
    : numberOfElements(weights.size()),
      probabilities(weights.size()),
      alias(weights.size()) {
    updateDistribution(weights);
}

void DiscreteDistribution::updateDistribution(const std::vector<float>& weights) {
    if (weights.size() != numberOfElements) {
        numberOfElements = weights.size();
        probabilities.resize(numberOfElements);
        alias.resize(numberOfElements);
    }
    
    if (numberOfElements == 0) return;

    // clear workspace
    small.clear();
    large.clear();

    // normalize weights
    float sum = 0.f;
    for (float w : weights) sum += w;
    for (size_t i = 0; i < numberOfElements; i++) probabilities[i] = weights[i] * static_cast<float>(numberOfElements) / sum;

    // create alias table
    for (size_t i = 0; i < numberOfElements; i++) {
        if (probabilities[i] < 1.0f) small.emplace_back(i);
        else large.emplace_back(i);
    }

    while(!small.empty() && !large.empty()) {
        const size_t s = small.back();
        const size_t l = large.back();
        small.pop_back();
        large.pop_back();
        
        alias[s] = l;
        probabilities[l] = probabilities[l] + probabilities[s] - 1.0f;

        if (probabilities[l] < 1.0f) small.emplace_back(l);
        else large.emplace_back(l);
    }
}