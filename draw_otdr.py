import os
import json
import matplotlib.pyplot as plt

def plot_sor(json_path: str):
    with open(json_path, "r") as f:
        raw = json.load(f)

    # Parse trace data points
    x, y = [], []
    for point in raw[2]:
        dist, power = point.strip().split("\t")
        x.append(float(dist))
        y.append(float(power))

    # Extract metadata & events
    meta      = raw[1]
    events    = meta["KeyEvents"]
    num_events = events["num events"]
    wavelength = meta["GenParams"]["wavelength"]
    fiber_id   = meta["GenParams"]["fiber ID"]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y, linewidth=0.8, color="steelblue", label="OTDR Trace")

    for i in range(1, num_events + 1):
        event = events[f"event {i}"]
        ex = float(event["distance"])
        ax.axvline(x=ex, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(ex, max(y) - 1, f"E{i}", fontsize=7, color="red", ha="center")

    ax.set_title(f"OTDR Trace — {fiber_id} @ {wavelength}")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # # Save plot next to the JSON file
    # plot_path = os.path.splitext(json_path)[0] + ".png"
    # plt.savefig(plot_path, dpi=150)
    # plt.close()
    # print(f"  Saved -> {plot_path}")


def plot_folder(folder: str):
    json_files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]

    if not json_files:
        print(f"No JSON files found in {folder}")
        return

    print(f"Found {len(json_files)} file(s). Plotting...")

    for filename in json_files:
        path = os.path.join(folder, filename)
        print(f"Processing: {filename}")
        try:
            plot_sor(path)
        except Exception as e:
            print(f"  Error: {e}")

    print("Done.")


if __name__ == "__main__":
    folder = "./output/parsed_folder"   # <-- change this
    plot_folder(folder)
