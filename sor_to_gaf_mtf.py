# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from pyts.image import GramianAngularField


# def load_trace(json_path):
#     with open(json_path) as f:
#         raw = json.load(f)

#     x, y = [], []
#     for point in raw[2]:
#         dist, power = point.strip().split("\t")
#         x.append(float(dist))
#         y.append(float(power))

#     # Get last event distance as cutoff
#     events = raw[1]["KeyEvents"]
#     last_event = events[f"event {events['num events']}"]
#     cutoff = float(last_event["distance"])

#     # Trim noise after last event
#     x, y = np.array(x), np.array(y)
#     mask = x <= cutoff
#     return y[mask]


# def to_gaf_image(signal, output_path, method):
#     signal = signal.reshape(1, -1)  # (n_samples, n_timestamps)
#     gaf = GramianAngularField(image_size=224, method=method)
#     image = gaf.fit_transform(signal)[0]

#     plt.figure(figsize=(4, 4))
#     plt.imshow(image, cmap="rainbow", origin="lower")
#     plt.axis("off")
#     plt.tight_layout(pad=0)
#     plt.savefig(output_path, dpi=150, bbox_inches="tight")
#     plt.close()


# def process_folder(input_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

#     print(f"Found {len(json_files)} file(s). Processing...")
#     for filename in json_files:
#         print(f"  Processing: {filename}")
#         try:
#             signal = load_trace(os.path.join(input_folder, filename))

#             base_name = filename.replace(".json", "")

#             out_sum = os.path.join(output_folder, f"{base_name}_sum.png")
#             out_diff = os.path.join(output_folder, f"{base_name}_diff.png")

#             to_gaf_image(signal, out_sum, method="summation")
#             to_gaf_image(signal, out_diff, method="difference")

#             print(f"  Saved -> {out_sum}")
#             print(f"  Saved -> {out_diff}")

#         except Exception as e:
#             print(f"  Error: {e}")

#     print("Done.")


# if __name__ == "__main__":
#     process_folder(
#         input_folder="./output/parsed_folder",
#         output_folder="./output/gaf_images"
#     )


# import os
# import json
# import numpy as np
# from pyts.image import GramianAngularField


# def load_trace(json_path):
#     with open(json_path) as f:
#         raw = json.load(f)

#     x, y = [], []
#     for point in raw[2]:
#         dist, power = point.strip().split("\t")
#         x.append(float(dist))
#         y.append(float(power))

#     # Get last event distance as cutoff
#     events = raw[1]["KeyEvents"]
#     last_event = events[f"event {events['num events']}"]
#     cutoff = float(last_event["distance"])

#     x, y = np.array(x), np.array(y)
#     mask = x <= cutoff
#     return y[mask]


# def to_gaf_tensor(signal):
#     signal = signal.reshape(1, -1)

#     gaf_sum = GramianAngularField(image_size=224, method="summation")
#     gaf_diff = GramianAngularField(image_size=224, method="difference")

#     img_sum = gaf_sum.fit_transform(signal)[0]
#     img_diff = gaf_diff.fit_transform(signal)[0]

#     # Stack into (2, H, W)
#     stacked = np.stack([img_sum, img_diff], axis=0)

#     return stacked.astype(np.float32)


# def process_folder(input_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

#     print(f"Found {len(json_files)} file(s). Processing...")
#     for filename in json_files:
#         print(f"  Processing: {filename}")
#         try:
#             signal = load_trace(os.path.join(input_folder, filename))

#             tensor = to_gaf_tensor(signal)

#             out_path = os.path.join(
#                 output_folder,
#                 filename.replace(".json", ".npy")
#             )

#             np.save(out_path, tensor)
#             print(f"  Saved -> {out_path} | shape = {tensor.shape}")

#         except Exception as e:
#             print(f"  Error: {e}")

#     print("Done.")


# if __name__ == "__main__":
#     process_folder(
#         input_folder="./output/parsed_folder",
#         output_folder="./output/gaf_tensors"
#     )




import os
import json
import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField


def load_trace(json_path):
    with open(json_path) as f:
        raw = json.load(f)

    x, y = [], []
    for point in raw[2]:
        dist, power = point.strip().split("\t")
        x.append(float(dist))
        y.append(float(power))

    events = raw[1]["KeyEvents"]
    last_event = events[f"event {events['num events']}"]
    cutoff = float(last_event["distance"])

    x, y = np.array(x), np.array(y)
    mask = x <= cutoff
    return y[mask]


def to_3channel_tensor(signal):
    signal = signal.reshape(1, -1)

    gaf_sum = GramianAngularField(image_size=224, method="summation")
    gaf_diff = GramianAngularField(image_size=224, method="difference")
    mtf = MarkovTransitionField(image_size=224)

    img_sum = gaf_sum.fit_transform(signal)[0]
    img_diff = gaf_diff.fit_transform(signal)[0]
    img_mtf = mtf.fit_transform(signal)[0]

    stacked = np.stack([img_sum, img_diff, img_mtf], axis=0)
    return stacked.astype(np.float32)


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    print(f"Found {len(json_files)} file(s). Processing...")
    for filename in json_files:
        print(f"  Processing: {filename}")
        try:
            signal = load_trace(os.path.join(input_folder, filename))
            tensor = to_3channel_tensor(signal)

            out_path = os.path.join(
                output_folder,
                filename.replace(".json", ".npy")
            )

            np.save(out_path, tensor)
            print(f"  Saved -> {out_path}, shape={tensor.shape}")

        except Exception as e:
            print(f"  Error: {e}")

    print("Done.")


if __name__ == "__main__":
    process_folder(
        input_folder="./output/parsed_folder",
        output_folder="./output/gaf_mtf_tensors"
    )