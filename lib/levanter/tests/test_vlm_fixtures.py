# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from lib.levanter.tests.vlm_fixtures import create_grid_mask, pad_pixel_values, verify_pixel_values_consistency


def test_image_tile_padding_preserves_pixels_and_mask() -> None:
    pixels = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    padded = pad_pixel_values(pixels, 4)
    mask = create_grid_mask(2, 4)
    np.testing.assert_array_equal(padded[:2], pixels)
    np.testing.assert_array_equal(padded[2:], np.zeros((2, 3, 2, 2)))
    np.testing.assert_array_equal(mask, [True, True, False, False])
    assert verify_pixel_values_consistency(pixels, padded, mask)
    padded[0, 0, 0, 0] += 1
    assert not verify_pixel_values_consistency(pixels, padded, mask)
