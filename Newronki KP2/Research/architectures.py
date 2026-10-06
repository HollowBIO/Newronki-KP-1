import numpy as np
from tensorflow import keras

IMG_SIZE = (150, 150)

# C<n> = Conv2D with n filters 3×3 + ReLU, P = MaxPooling2D 2×2.
# Everything else (input, Flatten, Dense(128), Dense(3), training) stays as in the base CNN.
# one_block / base / five_blocks vary the number of conv+pool blocks;
# pool_first / pool_last reorder the base's 3 conv + 3 pool layers;
# double_conv puts two convolutions before each pooling (VGG style);
# double_conv_pool uses blocks of two convolutions + two poolings.
VARIANTS = {
    "one_block": "C32-P",
    "base": "C32-P-C64-P-C128-P",
    "five_blocks": "C32-P-C64-P-C128-P-C128-P-C128-P",
    "pool_first": "P-C32-P-C64-P-C128",
    "pool_last": "C32-C64-C128-P-P-P",
    "double_conv": "C32-C32-P-C64-C64-P-C128-C128-P",
    "double_conv_pool": "C32-C32-P-P-C64-C64-P-P-C128-C128-P-P",
}


def build_model(spec):
    layers = [keras.layers.Input(shape=IMG_SIZE + (3,)), keras.layers.Rescaling(1.0 / 255)]
    for token in spec.split("-"):
        if token == "P":
            layers.append(keras.layers.MaxPooling2D())
        else:
            layers.append(keras.layers.Conv2D(int(token[1:]), 3, activation="relu"))
    layers += [
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ]
    return keras.Sequential(layers)


def flatten_size(model):
    return int(np.prod(model.layers[-3].output.shape[1:]))


def receptive_field(spec):
    """Width in input pixels of the patch one neuron of the last conv/pool layer sees."""
    size, jump = 1, 1
    for token in spec.split("-"):
        if token == "P":
            size += jump  # 2×2 window, stride 2
            jump *= 2
        else:
            size += 2 * jump  # 3×3 window, stride 1
    return size


if __name__ == "__main__":
    for name, spec in VARIANTS.items():
        model = build_model(spec)
        print(f"{name:<17} {spec:<40} params {model.count_params():>11,}  "
              f"flatten {flatten_size(model):>7,}  receptive field {receptive_field(spec):>3} px")
