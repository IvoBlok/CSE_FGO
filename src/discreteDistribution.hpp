#ifndef DISCRETE_DISTRIBUTION_H
#define DISCRETE_DISTRIBUTION_H

#include <vector>
#include <random>

class DiscreteDistribution {
public:
    DiscreteDistribution();
    ~DiscreteDistribution();
    
    void updateDistribution(std::vector<double>& weights);

    int generate(std::minstd_rand0& gen) const;

private:

};

#endif