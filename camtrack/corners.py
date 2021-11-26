#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli,
    filter_frame_corners
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))

# constants
maxCorners = 400
qualityLevel = 0.04

minDistance = 10
blockSize = 10
gradientSize = 31
useHarrisDetector = False

lk_params = dict(winSize=(8, 8),
                     maxLevel=7,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

glob_idx = 0

area = minDistance
size = 4

def find_corners(image_0, mindist=minDistance, grSize=gradientSize, mask=None):
    corners = cv2.goodFeaturesToTrack(image_0, maxCorners, qualityLevel, mindist, mask, \
            blockSize=blockSize, gradientSize=grSize, useHarrisDetector=useHarrisDetector)
    return None if corners is None else corners.reshape(-1, 2)

def find_corner_pyramid(image_0, positions, pyrs=2):
    ans_corners = np.empty((0, 2))
    sizes = np.array([])

    iterate_image = image_0.copy()
    mask = create_mask(positions, image_0)
    it = 1

    for _ in range(pyrs):
        corners = find_corners(iterate_image[::it,::it], mask=mask[::(it), ::(it)], mindist=minDistance / it)
        if not corners is None:
            corners *= it
            mask = create_mask(corners, image_0, mask, rad=area)

        ans_corners = np.concatenate((ans_corners, np.empty((0, 2)) if corners is None else corners))
        sizes = np.concatenate((sizes, np.array([(it+5) for t in range(len(corners) if not corners is None else 0)])))

        it *= 2
    
    return ans_corners, sizes



def create_mask(positions, image, mask=None, rad=area):
    if mask is None:
        mask = np.ones_like(image)

    for x, y in positions:
        if x < 0 or y < 0:
            continue

        left_lim = min(max(0, int(x) - rad // 2), image.shape[1])
        right_lim = max(min(image.shape[1], int(x) + rad // 2), 0)

        bottom_lim = min(max(0, int(y) - rad // 2), image.shape[0])
        top_lim = max(min(image.shape[0], int(y) + rad // 2), 0)

        zeros = np.zeros((top_lim - bottom_lim, right_lim - left_lim))

        mask[bottom_lim:top_lim, left_lim:right_lim] = zeros

    return mask.astype(np.uint8)

def to8bit(image):
        return (image * 255).astype(np.uint8)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    
    image = frame_sequence[0]


    corners, sizes = find_corner_pyramid(image, np.empty((0, 2)), pyrs=5)
    glob_idx = len(corners)
    idxs = np.arange(0, glob_idx)

    prev_frame_corners = FrameCorners(idxs, corners, sizes)

    for frame, image in enumerate(frame_sequence[1:], 1):
        positions, states, _ = cv2.calcOpticalFlowPyrLK(to8bit(frame_sequence[frame - 1]),
                                                        to8bit(image),
                                                        prev_frame_corners.points.astype('float32'),
                                                        None,
                                                        **lk_params)
        prev_frame_corners = filter_frame_corners(prev_frame_corners, states.ravel() == 1)
        builder.set_corners_at_frame(frame - 1, prev_frame_corners)

        survived_idxs = prev_frame_corners.ids
        survived_idxs = survived_idxs.reshape(-1)
        survived_positions = positions[states.ravel() == 1]

        mask = create_mask(survived_positions, image)
        
        corners, sizes = find_corner_pyramid(image, positions, pyrs=5)

        if corners is None:
            new_positions = survived_positions
            new_idxs = survived_idxs
        else:
            new_idxs = np.arange(glob_idx, glob_idx + len(corners))

            glob_idx = glob_idx + len(corners)

            new_positions = np.concatenate((survived_positions, corners))
            new_idxs = np.concatenate((survived_idxs, new_idxs))

        blocks = np.concatenate((prev_frame_corners.sizes.reshape(-1,), sizes))

        prev_frame_corners = FrameCorners(new_idxs, new_positions, blocks)
    builder.set_corners_at_frame(len(frame_sequence) - 1, prev_frame_corners)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
