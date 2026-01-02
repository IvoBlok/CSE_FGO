import matplotlib.pyplot as plt
import numpy as np

def plotSuccessData(data, samplesize, label, color=None):
    N_values = sorted(data.keys())
    success_counts = [data[n] for n in N_values]
    success_ratios = [count / samplesize for count in success_counts]

    N_zero = [N for N, r in zip(N_values, success_ratios) if r == 0]
    N_non_zero = [N for N, r in zip(N_values, success_ratios) if r > 0]
    ratios_non_zero = [r for r in success_ratios if r > 0]

    plt.semilogy(N_non_zero, ratios_non_zero, 'bo', markersize=5, label=label, 
                 linestyle='', color=color)
    plt.semilogy(N_zero, [1e-3] * len(N_zero), 'rx', markersize=7, 
                markeredgewidth=2, linestyle='', color=color)

success_data_paper_DMC1 = {
    50: 242, 51: 372, 52: 392, 53: 411, 54: 418, 55: 389, 56: 219, 57: 212, 58: 158, 59: 87,
    60: 262, 61: 220, 62: 114, 63: 257, 64: 248, 65: 47, 66: 146, 67: 91, 68: 151, 69: 271,
    70: 319, 71: 255, 72: 234, 73: 168, 74: 202, 75: 77, 76: 49, 77: 58, 78: 371, 79: 350,
    80: 282, 81: 93, 82: 17, 83: 1, 84: 16, 85: 271, 86: 7, 87: 41, 88: 197, 89: 75,
    90: 64, 91: 27, 92: 3, 93: 54, 94: 18, 95: 3, 96: 41, 97: 55, 98: 59, 99: 27,
    100: 15, 101: 8, 102: 20, 103: 31, 104: 12, 105: 19, 106: 16, 107: 44, 108: 18, 109: 8,
    110: 2, 111: 5, 112: 7, 113: 26, 114: 6, 115: 4, 116: 3, 117: 2, 118: 1, 119: 2
}

success_data_paper_DMC2 = {
    50: 423, 51: 444, 52: 426, 53: 483, 54: 458, 55: 434, 56: 454, 57: 442, 58: 458, 59: 428,
    60: 450, 61: 451, 62: 454, 63: 482, 64: 491, 65: 373, 66: 449, 67: 445, 68: 457, 69: 483,
    70: 476, 71: 438, 72: 447, 73: 431, 74: 424, 75: 111, 76: 87, 77: 94, 78: 420, 79: 397,
    80: 407, 81: 400, 82: 220, 83: 250, 84: 213, 85: 373, 86: 232, 87: 204, 88: 235, 89: 270,
    90: 253, 91: 255, 92: 262, 93: 251, 94: 280, 95: 288, 96: 264, 97: 261, 98: 67, 99: 278,
    100: 267, 101: 301, 102: 76, 103: 78, 104: 71, 105: 298, 106: 310, 107: 320, 108: 290, 109: 325,
    110: 305, 111: 297, 112: 348, 113: 316, 114: 327, 115: 316, 116: 349, 117: 323, 118: 330, 119: 332
}

success_data_paper_SMC = {
    50: 467, 51: 466, 52: 475, 53: 501, 54: 488, 55: 444, 56: 455, 57: 466, 58: 507, 59: 472,
    60: 479, 61: 478, 62: 503, 63: 503, 64: 507, 65: 373, 66: 476, 67: 445, 68: 457, 69: 483,
    70: 477, 71: 438, 72: 448, 73: 431, 74: 427, 75: 111, 76: 87, 77: 94, 78: 436, 79: 400,
    80: 407, 81: 400, 82: 220, 83: 250, 84: 213, 85: 373, 86: 232, 87: 204, 88: 235, 89: 270,
    90: 253, 91: 255, 92: 262, 93: 251, 94: 280, 95: 288, 96: 264, 97: 261, 98: 67, 99: 278,
    100: 267, 101: 301, 102: 76, 103: 78, 104: 71, 105: 298, 106: 310, 107: 320, 108: 290, 109: 325,
    110: 305, 111: 297, 112: 348, 113: 316, 114: 327, 115: 316, 116: 349, 117: 323, 118: 330, 119: 332
}

# the first success_data here comes from the main DMC implementation I've been using up to this point: 
"""
stepsSinceImprovement += 1;
if (deltaAtomEnergy < 0.0f || uniformDist(rng) < std::exp(-deltaAtomEnergy * dmcParams.invAcceptanceEnergy)) {
    localDiscreteOptimization(proposal);
    float candidateEnergy = proposal.getClusterEnergy(params.gridSpacingSquared); 
    
    if(candidateEnergy < state.discCandidates.back().second) { // the back is guaranteed to be the best
        state.discCandidates.emplace_back(proposal, candidateEnergy);
        stepsSinceImprovement = 0;
    }
    proposal.copyTo(walker);
}
"""
success_data_DMC1 = {0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 201, 7: 934, 8: 896, 9: 912, 10: 930, 11: 937, 12: 979, 13: 997, 14: 926, 15: 857, 16: 733, 17: 687, 18: 726, 19: 844, 20: 673, 21: 593, 22: 636, 23: 730, 24: 614, 25: 517, 26: 544, 27: 434, 28: 285, 29: 434, 30: 303, 31: 164, 32: 129, 33: 159, 34: 108, 35: 153, 36: 127, 37: 58, 38: 1, 39: 156, 40: 172, 41: 166, 42: 137, 43: 118, 44: 90, 45: 95, 46: 111, 47: 127, 48: 113, 49: 70, 50: 34, 51: 74, 52: 58, 53: 96, 54: 93, 55: 82, 56: 85, 57: 59, 58: 46, 59: 18, 60: 51, 61: 26, 62: 29, 63: 63, 64: 38, 65: 11, 66: 24, 67: 9, 68: 7, 69: 39}
success_data_DMC2 = {0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 200, 7: 964, 8: 931, 9: 961, 10: 966, 11: 971, 12: 988, 13: 1000, 14: 938, 15: 861, 16: 776, 17: 701, 18: 783, 19: 873, 20: 660, 21: 601, 22: 695, 23: 783, 24: 614, 25: 539, 26: 569, 27: 421, 28: 347, 29: 444, 30: 308, 31: 175, 32: 136, 33: 176, 34: 143, 35: 149, 36: 148, 37: 123, 38: 2, 39: 156, 40: 158, 41: 150, 42: 141, 43: 147, 44: 113, 45: 112, 46: 115, 47: 118, 48: 95, 49: 126, 50: 54, 51: 65, 52: 85, 53: 99, 54: 98, 55: 78, 56: 76, 57: 80, 58: 43, 59: 37, 60: 48, 61: 45, 62: 39, 63: 68, 64: 60, 65: 16, 66: 45, 67: 30, 68: 49, 69: 31}

# the second comes from my experimental rewrite, based on a new interpretation of the wording in the original FGO paper and the papers using it
"""
if (deltaAtomEnergy < 0.0f || uniformDist(rng) < std::exp(-deltaAtomEnergy * dmcParams.invAcceptanceEnergy)) {
    // accept move as new walker
    recomputeWeights = true;
    localDiscreteOptimization(proposal);
    proposal.copyTo(walker);

    float candidateEnergy = proposal.getClusterEnergy(params.gridSpacingSquared); 

    if(candidateEnergy < state.discCandidates.back().second) {
        state.discCandidates.emplace_back(proposal, candidateEnergy);
        stepsSinceImprovement = 0;
    } else {
        stepsSinceImprovement += 1;
    }
}
"""
success_data_DMC1_v2 = {0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 206, 7: 988, 8: 966, 9: 985, 10: 995, 11: 988, 12: 1000, 13: 1000, 14: 974, 15: 878, 16: 768, 17: 740, 18: 832, 19: 945, 20: 794, 21: 698, 22: 797, 23: 834, 24: 771, 25: 687, 26: 691, 27: 630, 28: 469, 29: 610, 30: 558, 31: 263, 32: 276, 33: 289, 34: 282, 35: 298, 36: 301, 37: 253, 38: 2, 39: 323, 40: 325, 41: 352, 42: 321, 43: 338, 44: 328, 45: 354, 46: 358, 47: 344, 48: 354, 49: 334, 50: 241, 51: 239, 52: 171, 53: 167, 54: 154, 55: 149, 56: 145, 57: 170, 58: 196, 59: 170, 60: 258, 61: 221, 62: 150, 63: 223, 64: 217, 65: 61, 66: 118, 67: 54, 68: 90, 69: 191}
success_data_DMC2_v2 = {0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 233, 7: 995, 8: 991, 9: 1000, 10: 999, 11: 1000, 12: 1000, 13: 1000, 14: 969, 15: 877, 16: 768, 17: 735, 18: 823, 19: 967, 20: 780, 21: 714, 22: 792, 23: 836, 24: 781, 25: 686, 26: 702, 27: 636, 28: 500, 29: 591, 30: 592, 31: 310, 32: 297, 33: 309, 34: 308, 35: 326, 36: 279, 37: 315, 38: 3, 39: 346, 40: 328, 41: 320, 42: 337, 43: 340, 44: 343, 45: 391, 46: 352, 47: 371, 48: 358, 49: 371, 50: 341, 51: 255, 52: 175, 53: 159, 54: 166, 55: 152, 56: 155, 57: 183, 58: 225, 59: 234, 60: 268, 61: 302, 62: 204, 63: 267, 64: 262, 65: 80, 66: 157, 67: 138, 68: 164, 69: 207}
samplesize = 1000

plt.figure(figsize=(12, 7))

# Choose a color palette
colors = plt.cm.tab10.colors  # or any other colormap
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Tab10 colors
plotSuccessData(success_data_paper_DMC1, samplesize, "DMC1 Paper", color=colors[0])
plotSuccessData(success_data_paper_DMC2, samplesize, "DMC2 Paper", color=colors[1])

plotSuccessData(success_data_DMC1, samplesize, "DMC1 v1", color=colors[2])
plotSuccessData(success_data_DMC2, samplesize, "DMC2 v1", color=colors[3])

plotSuccessData(success_data_DMC1_v2, samplesize, "DMC1 v2", color=colors[4])
plotSuccessData(success_data_DMC2_v2, samplesize, "DMC2 v2", color=colors[5])

plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)

plt.xlabel('N', fontsize=14)
plt.ylabel('Success Ratio', fontsize=14)
plt.title(f'Success Ratio vs. N (Successes / {samplesize} attempts)', fontsize=16, fontweight='bold')
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()