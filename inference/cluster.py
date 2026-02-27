
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ----------------------------
# Load embeddings
# ----------------------------
embeddings = np.load("otdr_embeddings.npy")

tensor_folder = "./output/parsed_folder"  # folder with OTDR JSONs

# Get JSON filenames sorted (assumes 1-to-1 correspondence with embeddings)
filenames = sorted([f for f in os.listdir(tensor_folder) if f.endswith(".json")])

if len(filenames) != len(embeddings):
    raise ValueError(f"Number of embeddings ({len(embeddings)}) does not match number of JSON files ({len(filenames)})")

# ----------------------------
# DBSCAN clustering
# ----------------------------
dbscan = DBSCAN(eps=0.01, min_samples=2, metric='euclidean')
labels = dbscan.fit_predict(embeddings)
clusters = set(labels)

print(f"Found {len(clusters) - (1 if -1 in labels else 0)} clusters (excluding outliers)")
print(f"Outliers: {list(labels).count(-1)}")

# ----------------------------
# Helper to load OTDR trace
# ----------------------------
def load_otdr_trace(json_path):
    with open(json_path) as f:
        raw = json.load(f)
    x, y = [], []
    for point in raw[2]:
        dist, power = point.strip().split("\t")
        x.append(float(dist))
        y.append(float(power))
    # Trim at last event
    events = raw[1]["KeyEvents"]
    last_event = events[f"event {events['num events']}"]
    cutoff = float(last_event["distance"])
    x, y = np.array(x), np.array(y)
    mask = x <= cutoff
    return x[mask], y[mask]

# ----------------------------
# Plot OTDR traces per cluster with file names
# ----------------------------
for cluster_label in clusters:
    plt.figure(figsize=(14,6))
    
    cluster_indices = np.where(labels == cluster_label)[0]
    
    if cluster_label == -1:
        plt.title(f"DBSCAN Outliers (Anomalous fibers) - {len(cluster_indices)} traces")
        color = "red"
    else:
        plt.title(f"Cluster {cluster_label} - {len(cluster_indices)} traces")
        color = "blue"
    
    for idx in cluster_indices:
        json_path = os.path.join(tensor_folder, filenames[idx])
        
        if not os.path.exists(json_path):
            print(f"Warning: JSON not found -> {json_path}")
            continue
        
        dist, trace = load_otdr_trace(json_path)
        plt.plot(dist, trace, color=color, alpha=0.5, linewidth=0.8)
        
        # Annotate filename near the end of trace
        plt.text(dist[-1], trace[-1], filenames[idx], fontsize=6, rotation=30, alpha=0.7)
    
    plt.xlabel("Distance (km)")
    plt.ylabel("Power (dB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()