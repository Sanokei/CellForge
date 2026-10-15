# SPDX-License-Identifier: MIT-0
"""Original declared-family coordinate oracle; no ARC query labels."""
import numpy as np

def mirror_example(seed):
    rng = np.random.default_rng(seed)
    height, width = map(int, rng.integers(6, 31, size=2))
    axis = int(rng.integers(2))
    direction = int(rng.choice([-1, 1]))
    size, cross_size = (height, width) if axis == 0 else (width, height)
    pivot = int(rng.integers(2, size - 2))
    cross = int(rng.integers(1, cross_size - 1))
    color = int(rng.choice([1, 3, 4, 5, 6, 7, 8, 9]))
    image = np.zeros((height, width), dtype=int)
    points = {(pivot + direction, cross)}
    for depth in (1, 2):
        for offset in (-1, 0, 1):
            if rng.random() < .6:
                points.add((pivot + direction * depth, cross + offset))
    def coordinates(along, across):
        return (along, across) if axis == 0 else (across, along)
    image[coordinates(pivot, cross)] = 2
    expected = np.zeros_like(image)
    for along, across in points:
        image[coordinates(along, across)] = color
        expected[coordinates(along, across)] = color
        reflected = 2 * pivot + direction - along
        expected[coordinates(reflected, across)] = color
    return image.tolist(), expected.tolist()


def generate_marker_task(seed):
    pairs=[]
    for i in range(3):
        image, output=mirror_example(seed*3+i)
        pairs.append(dict(input=image,output=output))
    return dict(family="marker_mirror",train=pairs[:2],test=pairs[2:])
