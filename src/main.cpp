#include <iostream>
#include <fstream>

#include "json.hpp"
#include "FGO.hpp"

const float clusterBestEnergies[151] = {
    0.f,  // Placeholder for index 0 (no cluster with 0 atoms)
    0.f,  // Placeholder for index 1 (no cluster with 1 atom)
    -1.000000f,    // 2 atoms
    -3.000000f,    // 3 atoms
    -6.000000f,    // 4 atoms
    -9.103852f,    // 5 atoms
    -12.712062f,   // 6 atoms
    -16.505384f,   // 7 atoms
    -19.821489f,   // 8 atoms
    -24.113360f,   // 9 atoms
    -28.422532f,   // 10 atoms
    -32.765970f,   // 11 atoms
    -37.967600f,   // 12 atoms
    -44.326801f,   // 13 atoms
    -47.845157f,   // 14 atoms
    -52.322627f,   // 15 atoms
    -56.815742f,   // 16 atoms
    -61.317995f,   // 17 atoms
    -66.530949f,   // 18 atoms
    -72.659782f,   // 19 atoms
    -77.177043f,   // 20 atoms
    -81.684571f,   // 21 atoms
    -86.809782f,   // 22 atoms
    -92.844472f,   // 23 atoms
    -97.348815f,   // 24 atoms
    -102.372663f,  // 25 atoms
    -108.315616f,  // 26 atoms
    -112.873584f,  // 27 atoms
    -117.822402f,  // 28 atoms
    -123.587371f,  // 29 atoms
    -128.286571f,  // 30 atoms
    -133.586422f,  // 31 atoms
    -139.635524f,  // 32 atoms
    -144.842719f,  // 33 atoms
    -150.044528f,  // 34 atoms
    -155.756643f,  // 35 atoms
    -161.825363f,  // 36 atoms
    -167.033672f,  // 37 atoms
    -173.928427f,  // 38 atoms
    -180.033185f,  // 39 atoms
    -185.249839f,  // 40 atoms
    -190.536277f,  // 41 atoms
    -196.277534f,  // 42 atoms
    -202.364664f,  // 43 atoms
    -207.688728f,  // 44 atoms
    -213.784862f,  // 45 atoms
    -220.680330f,  // 46 atoms
    -226.012256f,  // 47 atoms
    -232.199529f,  // 48 atoms
    -239.091864f,  // 49 atoms
    -244.549926f,  // 50 atoms
    -251.253964f,  // 51 atoms
    -258.229991f,  // 52 atoms
    -265.203016f,  // 53 atoms
    -272.208631f,  // 54 atoms
    -279.248470f,  // 55 atoms
    -283.643105f,  // 56 atoms
    -288.342625f,  // 57 atoms
    -294.378148f,  // 58 atoms
    -299.738070f,  // 59 atoms
    -305.875476f,  // 60 atoms
    -312.008896f,  // 61 atoms
    -317.353901f,  // 62 atoms
    -323.489734f,  // 63 atoms
    -329.620147f,  // 64 atoms
    -334.971532f,  // 65 atoms
    -341.110599f,  // 66 atoms
    -347.252007f,  // 67 atoms
    -353.394542f,  // 68 atoms
    -359.882566f,  // 69 atoms
    -366.892251f,  // 70 atoms
    -373.349661f,  // 71 atoms
    -378.637253f,  // 72 atoms
    -384.789377f,  // 73 atoms
    -390.908500f,  // 74 atoms
    -397.492331f,  // 75 atoms
    -402.894866f,  // 76 atoms
    -409.083517f,  // 77 atoms
    -414.794401f,  // 78 atoms
    -421.810897f,  // 79 atoms
    -428.083564f,  // 80 atoms
    -434.343643f,  // 81 atoms
    -440.550425f,  // 82 atoms
    -446.924094f,  // 83 atoms
    -452.657214f,  // 84 atoms
    -459.055799f,  // 85 atoms
    -465.384493f,  // 86 atoms
    -472.098165f,  // 87 atoms
    -479.032630f,  // 88 atoms
    -486.053911f,  // 89 atoms
    -492.433908f,  // 90 atoms
    -498.811060f,  // 91 atoms
    -505.185309f,  // 92 atoms
    -510.877688f,  // 93 atoms
    -517.264131f,  // 94 atoms
    -523.640211f,  // 95 atoms
    -529.879146f,  // 96 atoms
    -536.681383f,  // 97 atoms
    -543.665361f,  // 98 atoms
    -550.666526f,  // 99 atoms
    -557.039820f,  // 100 atoms
    -563.411308f,  // 101 atoms
    -569.363652f,  // 102 atoms
    -575.766131f,  // 103 atoms
    -582.086642f,  // 104 atoms
    -588.266501f,  // 105 atoms
    -595.061072f,  // 106 atoms
    -602.007110f,  // 107 atoms
    -609.033011f,  // 108 atoms
    -615.411166f,  // 109 atoms
    -621.788224f,  // 110 atoms
    -628.068416f,  // 111 atoms
    -634.874626f,  // 112 atoms
    -641.794704f,  // 113 atoms
    -648.833100f,  // 114 atoms
    -655.756307f,  // 115 atoms
    -662.809353f,  // 116 atoms
    -668.282701f,  // 117 atoms
    -674.769635f,  // 118 atoms
    -681.419158f,  // 119 atoms
    -687.021982f,  // 120 atoms
    -693.819577f,  // 121 atoms
    -700.939379f,  // 122 atoms
    -707.802109f,  // 123 atoms
    -714.920896f,  // 124 atoms
    -721.303235f,  // 125 atoms
    -727.349853f,  // 126 atoms
    -734.479629f,  // 127 atoms
    -741.332100f,  // 128 atoms
    -748.460647f,  // 129 atoms
    -755.271073f,  // 130 atoms
    -762.441558f,  // 131 atoms
    -768.042203f,  // 132 atoms
    -775.023203f,  // 133 atoms
    -782.206157f,  // 134 atoms
    -790.278120f,  // 135 atoms
    -797.453259f,  // 136 atoms
    -804.631473f,  // 137 atoms
    -811.812780f,  // 138 atoms
    -818.993848f,  // 139 atoms
    -826.174676f,  // 140 atoms
    -833.358586f,  // 141 atoms
    -840.538610f,  // 142 atoms
    -847.721698f,  // 143 atoms
    -854.904499f,  // 144 atoms
    -862.087012f,  // 145 atoms
    -869.272573f,  // 146 atoms
    -876.461207f,  // 147 atoms
    -881.072971f,  // 148 atoms
    -886.693405f,  // 149 atoms
    -893.310258f   // 150 atoms
};

using json = nlohmann::json;

json create_points_json(size_t n, const std::vector<float>& xs, const std::vector<float>& ys, const std::vector<float>& zs) {
    json points_array = json::array();
    
    for (size_t i = 0; i < n; i++) {
        points_array.push_back({
            {"x", xs[i]},
            {"y", ys[i]},
            {"z", zs[i]}
        });
    }
    
    return points_array;
}

void to_json(json& j, const Cluster& cluster) {
    j = json{{"points", create_points_json(cluster.size(), cluster.x, cluster.y, cluster.z)}};
}

void to_json(json& j, const SingleRunResult& result) {
    j = json{
        {"bestCluster", result.bestCluster},
        {"bestEnergy", result.bestEnergy},
        {"candidates", result.candidates},
        {"totalTime", result.totalTime.count()},
        {"dmc1Time", result.dmc1Time.count()},
        {"dmc2Time", result.dmc2Time.count()},
        {"realOptTime", result.realOptTime.count()}
    };
}

void to_json(json& j, const MultiRunResult& result) {
    j = json{
        {"n", result.globalBestCluster.n},
        {"globalBestCluster", result.globalBestCluster},
        {"globalBestEnergy", result.globalBestEnergy},
        {"allRuns", result.allRuns},
        {"totalTime", result.totalTime.count()},
        {"averageTime", result.averageTime.count()}
    };
}

struct FullBenchmarkResult {
    struct NResult {
        MultiRunResult multiRunResult;
        float exactSolution;
    };
    
    std::vector<NResult> allNResults;
    std::chrono::microseconds totalBenchmarkTime{0};
    std::string timestamp;
};

void to_json(json& j, const FullBenchmarkResult::NResult& nResult) {
    j = json{
        {"multiRunResult", nResult.multiRunResult},
        {"exactSolution", nResult.exactSolution}
    };
}

void to_json(json& j, const FullBenchmarkResult& benchmark) {
    j = json{
        {"allNResults", benchmark.allNResults},
        {"totalBenchmarkTime", benchmark.totalBenchmarkTime.count()},
        {"timestamp", benchmark.timestamp}
    };
}


int main(int argc, char** argv) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    FullBenchmarkResult benchmarkResult;
    
    const int NUM_RUNS = 100;
    const int MIN_N = 2;
    const int MAX_N = 65;
    
    for (int n = MIN_N; n <= MAX_N; n++) {
        std::cout << "Testing N = " << n << "..." << std::endl;
        
        FGOParameters params;
        params.numberOfAtoms = n;
        
        FuzzyGlobalOptimizer optimizer(params);
        
        auto multiResult = optimizer.runMultiple(NUM_RUNS);
        
        int correctFinds = 0;
        for (const auto& run : multiResult.allRuns) {
            if (std::abs(run.bestEnergy - clusterBestEnergies[n]) < 1e-3 || 
                run.bestEnergy < clusterBestEnergies[n]) {
                correctFinds++;
            }
        }
        
        FullBenchmarkResult::NResult nResult;
        nResult.multiRunResult = std::move(multiResult);
        nResult.exactSolution = clusterBestEnergies[n];
        benchmarkResult.allNResults.push_back(nResult);
        
        std::cout << "  Success rate: " << correctFinds << "/" << NUM_RUNS << "\n";
        std::cout << "  Avg time: " << nResult.multiRunResult.averageTime.count() / 1e6 << " seconds\n";
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    benchmarkResult.totalBenchmarkTime = 
        std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    // add timestamp
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    benchmarkResult.timestamp = oss.str();
    
    // save to JSON file
    json j = benchmarkResult;
    std::ofstream file("benchmark_results.json");
    file << j.dump(2);
    
    std::cout << "\nBenchmark completed in " << benchmarkResult.totalBenchmarkTime.count() / 1e6 << " seconds\n";
    std::cout << "Results saved to benchmark_results.json\n";
    
    return 0;
}