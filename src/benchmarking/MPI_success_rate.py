import matplotlib.pyplot as plt
import numpy as np

# success data from parallel run of FGO
success_data = {
    0: 200, 1: 200, 2: 200, 3: 200, 4: 200, 5: 200, 6: 36, 7: 188, 8: 190,
    9: 192, 10: 192, 11: 187, 12: 200, 13: 200, 14: 188, 15: 177, 16: 145,
    17: 134, 18: 158, 19: 173, 20: 132, 21: 124, 22: 147, 23: 154, 24: 110,
    25: 110, 26: 106, 27: 77, 28: 69, 29: 83, 30: 66, 31: 44, 32: 24, 33: 37,
    34: 22, 35: 33, 36: 26, 37: 18, 38: 0, 39: 37, 40: 33, 41: 26, 42: 27,
    43: 23, 44: 26, 45: 23, 46: 24, 47: 20, 48: 26, 49: 22, 50: 13, 51: 20,
    52: 14, 53: 20, 54: 27, 55: 20, 56: 20, 57: 11, 58: 10, 59: 3, 60: 11,
    61: 9, 62: 6, 63: 19, 64: 11, 65: 1, 66: 11, 67: 3, 68: 5, 69: 5, 70: 9,
    71: 6, 72: 11, 73: 8, 74: 10, 75: 0, 76: 0, 77: 3, 78: 7, 79: 11, 80: 11,
    81: 5, 82: 1, 83: 0, 84: 2, 85: 4, 86: 0, 87: 0, 88: 2, 89: 1, 90: 4,
    91: 1, 92: 0, 93: 2, 94: 1, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0,
    101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0
}

N_values = sorted(success_data.keys())
success_counts = [success_data[n] for n in N_values]
success_ratios = [count / 200.0 for count in success_counts]

plt.figure(figsize=(12, 7))

# separate zero and non-zero points
zero_mask = [r == 0 for r in success_ratios]
non_zero_mask = [r > 0 for r in success_ratios]

N_zero = [N for N, r in zip(N_values, success_ratios) if r == 0]
ratios_zero = [r for r in success_ratios if r == 0]

N_non_zero = [N for N, r in zip(N_values, success_ratios) if r > 0]
ratios_non_zero = [r for r in success_ratios if r > 0]

# plot non-zero points as blue circles
plt.semilogy(N_non_zero, ratios_non_zero, 'bo', markersize=6, label='Success > 0', linestyle='')

zero_y_value = 1e-4  # arbitrary small value that will show on log scale
plt.semilogy(N_zero, [zero_y_value] * len(N_zero), 'rx', markersize=8, 
             markeredgewidth=2, label='Zero Success', linestyle='')

plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

plt.xlabel('N', fontsize=14)
plt.ylabel('Success Ratio', fontsize=14)
plt.title('Success Ratio vs. N (Successes / 200 attempts)', fontsize=16, fontweight='bold')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xticks(np.arange(0, max(N_values)+1, 5))
plt.ylim([1e-5, 1.1])

plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()