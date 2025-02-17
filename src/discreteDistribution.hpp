#ifndef DISCRETE_DISTRIBUTION_H
#define DISCRETE_DISTRIBUTION_H

#include <vector>
#include <random>
#include <stdexcept>

class DiscreteDistribution {
public:
    DiscreteDistribution();
    DiscreteDistribution(int size);
    DiscreteDistribution(std::vector<float>& weights);

    void updateDistribution(std::vector<float>& weights);

    int generate(std::minstd_rand0& gen);

private:
    std::vector<int> alias;
    std::vector<float> probabilities;
    int numberOfElements;
    
    std::vector<int> small, large;

    std::uniform_real_distribution<> uniformDistribution;
};

#endif