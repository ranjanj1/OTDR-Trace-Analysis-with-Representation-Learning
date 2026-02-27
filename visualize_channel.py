import numpy as np
import json
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField, MarkovTransitionField

# ---- function to load OTDR signal + events ----
def load_otdr_signal_with_events(json_path):
    with open(json_path) as f:
        raw = json.load(f)

    # Trace data
    x, y = [], []
    for point in raw[2]:
        dist, power = point.strip().split("\t")
        x.append(float(dist))
        y.append(float(power))

    x = np.array(x)
    y = np.array(y)

    # Metadata & events
    meta = raw[1]
    events = meta["KeyEvents"]
    num_events = events["num events"]
    event_positions = [float(events[f"event {i}"]["distance"]) for i in range(1, num_events + 1)]
    fiber_id = meta["GenParams"]["fiber ID"]
    wavelength = meta["GenParams"]["wavelength"]

    # Cutoff at last event
    cutoff = event_positions[-1]
    mask = x <= cutoff
    x, y = x[mask], y[mask]

    return x, y, event_positions, fiber_id, wavelength

# ---- load tensor ----
tensor_path = "output/gaf_mtf_tensors/2022_06_01_1310_100ns_10sec_no_1.npy"
json_path = "output/parsed_folder/2022_06_01_1310_100ns_10sec_no_1.json"

tensor = np.load(tensor_path)  # (3, 224, 224)
dist, signal, event_positions, fiber_id, wavelength = load_otdr_signal_with_events(json_path)

titles = ["GAF Summation", "GAF Difference", "MTF"]

# ---- plot ----
plt.figure(figsize=(14, 8))

# Top: OTDR trace with events
plt.subplot(2, 1, 1)
plt.plot(dist, signal, linewidth=0.8, color="steelblue", label="OTDR Trace")
for i, ex in enumerate(event_positions, 1):
    plt.axvline(x=ex, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    plt.text(ex, max(signal)-1, f"E{i}", fontsize=8, color="red", ha="center")

plt.title(f"OTDR Trace — {fiber_id} @ {wavelength}")
plt.xlabel("Distance (km)")
plt.ylabel("Power (dB)")
plt.grid(True, alpha=0.3)
plt.legend()

# Bottom: tensor channels
for i in range(3):
    plt.subplot(2, 3, i + 4)  # bottom row, 3 cols
    plt.imshow(tensor[i], cmap="viridis", origin="lower")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()