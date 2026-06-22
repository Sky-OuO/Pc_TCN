import numpy as np
import json


labels = np.load('params/labels.npy')
bins = [-8, -7, -6, -5, -4, -3, -2, -1, 0]

bin_counts = {f"[1e{bins[i]:+.0f}, 1e{bins[i+1]:+.0f})": 0 for i in range(len(bins)-1)}
for label in labels:
    log_label = np.log10(label)
    for i in range(len(bins)-1):
        if bins[i] <= log_label < bins[i+1]:
            bin_counts[f"[1e{bins[i]:+.0f}, 1e{bins[i+1]:+.0f})"] += 1
            break

print(json.dumps(bin_counts, indent=2))
print(f"Total samples: {len(labels)}")