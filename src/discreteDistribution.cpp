#include "discreteDistribution.hpp"

DiscreteDistribution::DiscreteDistribution() { }

DiscreteDistribution::DiscreteDistribution(int size) {
    uniformDistribution = std::uniform_real_distribution<>(0.0, 1.0);

    numberOfElements = size;
    std::vector<float> probabilities(numberOfElements);
    std::vector<int> alias(numberOfElements);
    
    this->probabilities = std::move(probabilities);
    this->alias = std::move(alias);
}

DiscreteDistribution::DiscreteDistribution(std::vector<float>& weights) {
    uniformDistribution = std::uniform_real_distribution<>(0.0, 1.0);

    numberOfElements = weights.size();
    std::vector<float> probabilities(numberOfElements);
    std::vector<int> alias(numberOfElements);
    
    this->probabilities = std::move(probabilities);
    this->alias = std::move(alias);

    updateDistribution(weights);
}

void DiscreteDistribution::updateDistribution(std::vector<float>& weights) {
    if (numberOfElements != weights.size())
        throw std::runtime_error("distribution update weights is not of equal length as constructor weights!");

    // normalize weights
    float sum = 0.f;
    for (float w : weights) sum += w;
    for (int i = 0; i < numberOfElements; i++) probabilities[i] = weights[i] * numberOfElements / sum;

    // create alias table
    std::vector<int> small, large;
    for (int i = 0; i < numberOfElements; i++) {
        if (probabilities[i] < 1.f) small.push_back(i);
        else large.push_back(i);
    }

    while(!small.empty() && !large.empty()) {
        int s = small.back(), l = large.back();
        small.pop_back();
        large.pop_back();

        alias[s] = l;
        probabilities[l] = probabilities[l] + probabilities[s] - 1.f;

        if (probabilities[l] < 1.f) small.push_back(l);
        else large.push_back(l);
    }
}

int DiscreteDistribution::generate(std::minstd_rand0& gen) {
    int i = static_cast<int>(uniformDistribution(gen) * probabilities.size());
    return (uniformDistribution(gen) < probabilities[i]) ? i : alias[i];
}