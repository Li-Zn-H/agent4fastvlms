"""Image-only native InternVL transform for isolated quantizer processes.

The ordering and tie behavior match lmms-eval 0.7.3's internvl3.load_image.
Keep comparison tests against that pinned implementation; no size/tiling policy
is exposed here as an optimization setting.
"""


def native_grid(width, height, input_size=448, max_num=12):
    """Return the native LMMS tile grid, including its set iteration ties."""
    ratios = {(i, j) for n in range(1, max_num + 1)
              for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num}
    # Native LMMS converts the area-sorted ratios back to a set before choosing
    # a grid. Preserving that operation matters on equal-aspect-ratio ties.
    candidates = set(sorted(ratios, key=lambda pair: pair[0] * pair[1]))
    best, difference = (1, 1), float("inf")
    for pair in candidates:
        delta = abs(width / height - pair[0] / pair[1])
        if delta < difference or (delta == difference and
                                 width * height > .5 * input_size ** 2 * pair[0] * pair[1]):
            best, difference = pair, delta
    return best


def load_image(image, input_size=448, max_num=12):
    import torch
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    columns, rows = native_grid(*image.size, input_size, max_num)
    resized = image.resize((columns * input_size, rows * input_size))
    tiles = [resized.crop((x * input_size, y * input_size,
                           (x + 1) * input_size, (y + 1) * input_size))
             for y in range(rows) for x in range(columns)]
    if len(tiles) > 1:
        tiles.append(image.resize((input_size, input_size)))
    transform = T.Compose([
        T.Lambda(lambda value: value.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
    ])
    return torch.stack([transform(tile) for tile in tiles])
