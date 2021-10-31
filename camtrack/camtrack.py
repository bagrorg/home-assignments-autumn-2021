#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2
import sortednp as snp

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    TriangulationParameters,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    check_baseline
)

range_of_neighbours = 75

triang_params = TriangulationParameters(
    max_reprojection_error=6,
    min_triangulation_angle_deg=1,
    min_depth=0.11)
iterations = 108
baseline_min_dist = 0

def get_neighbours(i, frames):
    l = max(0, i - range_of_neighbours // 2)
    r = min(frames, i + range_of_neighbours // 2)

    res = set(range(l, r))
    return res


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    ###  INIT PART  ###
    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    point_cloud_builder = PointCloudBuilder()


    known_ind1 = known_view_1[0]
    known_ind2 = known_view_2[0]

    view_mat1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat2 = pose_to_view_mat3x4(known_view_2[1])

    view_mats[known_ind1] = view_mat1
    view_mats[known_ind2] = view_mat2

    frame1 = corner_storage[known_ind1]
    frame2 = corner_storage[known_ind2]
    
    correspondence = build_correspondences(frame1, frame2)
    pts3d, ids, cos_med = triangulate_correspondences(correspondence,
                                                        view_mat1,
                                                        view_mat2,
                                                        intrinsic_mat,
                                                        triang_params)

    point_cloud_builder.add_points(ids, pts3d)
    
    used = [False] * frame_count
    used[known_ind1] = True
    used[known_ind2] = True

    good_for_pnp = get_neighbours(known_ind1, frame_count) | get_neighbours(known_ind2, frame_count)

    ## MAIN LOOP ##
    while True:
        best_for_pnp = -1
        inliers_for_best = None
        r_vec_for_best = None
        t_vec_for_best = None
        ind1_for_best = None
        ind2_for_best = None
        corn_for_best = None
        outliers_for_best = None

        for i in good_for_pnp:
            if used[i]:
                continue

            corn = corner_storage[i]
            inds, (ind1, ind2) = snp.intersect(point_cloud_builder.ids.flatten(), 
                                               corn.ids.flatten(), indices=True)

            if len(inds) < 4:
                continue

            succ, r_vec, t_vec, inliers = cv2.solvePnPRansac(
                point_cloud_builder.points[ind1],
                corn.points[ind2],
                intrinsic_mat,
                np.array([]),
                iterationsCount=iterations,
                reprojectionError=triang_params.max_reprojection_error,
                flags=cv2.SOLVEPNP_EPNP
            )

            if succ:
                if best_for_pnp == -1:
                    inliers_for_best = inliers
                    r_vec_for_best = r_vec
                    t_vec_for_best = t_vec
                    best_for_pnp = i
                    ind1_for_best = ind1
                    ind2_for_best = ind2
                    corn_for_best = corn

                    mask = np.zeros(len(inds), dtype=bool)
                    mask[inliers.flatten()] = True
                    outliers_for_best = inds[~mask]

                elif len(inliers_for_best) < len(inliers):
                    inliers_for_best = inliers
                    r_vec_for_best = r_vec
                    t_vec_for_best = t_vec
                    best_for_pnp = i
                    ind1_for_best = ind1
                    ind2_for_best = ind2
                    corn_for_best = corn

                    mask = np.zeros(len(inds), dtype=bool)
                    mask[inliers.flatten()] = True
                    outliers_for_best = inds[~mask]

        if best_for_pnp == -1:
            break  

        succ, r_vec, t_vec = cv2.solvePnP(
            point_cloud_builder.points[ind1_for_best[inliers_for_best.flatten()]],
            corn_for_best.points[ind2_for_best[inliers_for_best.flatten()]],
            intrinsic_mat,
            np.array([]),
            r_vec_for_best,
            t_vec_for_best,
            True,
            cv2.SOLVEPNP_ITERATIVE
        )

        if not succ:
            continue

        view_mats[best_for_pnp] = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)

        best_pts3d = None
        best_ids = None
        cos_med_best = 1
        ns = get_neighbours(best_for_pnp, frame_count)
        for j in ns:
            if not used[j]:
                continue
            
            if not check_baseline(view_mats[j], view_mats[best_for_pnp], baseline_min_dist):
                continue

            correspondence = build_correspondences(corner_storage[best_for_pnp], corner_storage[j], outliers_for_best)
            pts3d, ids, cos_med = triangulate_correspondences(correspondence,
                                                                view_mats[best_for_pnp],
                                                                view_mats[j],
                                                                intrinsic_mat,
                                                                triang_params)

            if cos_med < cos_med_best:
                best_pts3d = pts3d
                best_ids = ids
        if best_pts3d is not None:
            point_cloud_builder.add_points(best_ids, best_pts3d)

        good_for_pnp |= ns
        used[best_for_pnp] = True
        

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
