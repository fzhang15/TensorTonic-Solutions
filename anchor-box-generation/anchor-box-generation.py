import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride = image_size / feature_size
    grid = np.arange(feature_size)
    grid_x, grid_y = np.meshgrid(grid, grid)
    cx = (grid_x + 0.5) * stride
    cy = (grid_y + 0.5) * stride
    centers = np.stack([cx.flatten(), cy.flatten()], axis=1)

    scales, aspect_ratios = np.asarray(scales), np.asarray(aspect_ratios)
    w = scales.reshape(-1, 1) * np.sqrt(aspect_ratios.reshape(1, -1))
    h = scales.reshape(-1, 1) / np.sqrt(aspect_ratios.reshape(1, -1))
    sizes = np.stack([w.flatten(), h.flatten()], axis=1)

    c = centers[:, np.newaxis, :]
    s = sizes[np.newaxis, :, :]

    top_left = c - s / 2
    bottom_right = c + s / 2

    boxes = np.concatenate([top_left, bottom_right], axis=-1)

    return boxes.reshape(-1, 4).tolist()