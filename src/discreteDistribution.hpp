#ifndef DISCRETE_DISTRIBUTION_H
#define DISCRETE_DISTRIBUTION_H

#include <vector>
#include <random>
#include <stdexcept>
#include <cstddef>

class DiscreteDistribution {
public:
    DiscreteDistribution() = default;
    explicit DiscreteDistribution(size_t size);
    explicit DiscreteDistribution(const std::vector<float>& weights);

    void updateDistribution(const std::vector<float>& weights);

    template<typename Generator>
    size_t generate(Generator& gen);

private:
    std::vector<size_t> alias;
    std::vector<float> probabilities;
    size_t numberOfElements = 0;
    
    std::vector<size_t> small, large;
};

template<typename Generator>
size_t DiscreteDistribution::generate(Generator& gen) {
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    const size_t i = static_cast<size_t>(uniform(gen) * numberOfElements);
    return (uniform(gen) < probabilities[i]) ? i : alias[i];
}

#endif