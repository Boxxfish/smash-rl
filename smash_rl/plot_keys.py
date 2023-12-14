"""
Creates a scatterplot of keys from the experience encoder.
"""

from matplotlib import pyplot as plt  # type: ignore
from pathlib import Path
import numpy as np

gen_path = Path("temp/generated")
all_keys_v_l = []
all_keys_p_l = []
for p in gen_path.iterdir():
    if f"v_keys" in p.name:
        keys = np.load(p)
        all_keys_v_l.append(keys)
    if f"p_keys" in p.name:
        keys = np.load(p)
        all_keys_p_l.append(keys)
all_keys_p = np.concatenate(all_keys_p_l, 0)
all_keys_p = all_keys_p.swapaxes(0, 1)  # Shape: (3, num_keys)
all_keys_v = np.concatenate(all_keys_v_l, 0)
all_keys_v = all_keys_v.swapaxes(0, 1)  # Shape: (3, num_keys)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(all_keys_p[0], all_keys_p[1], all_keys_p[2])
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.scatter(all_keys_v[0], all_keys_v[1], all_keys_v[2])
plt.show()