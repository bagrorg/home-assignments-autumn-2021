#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2
import sortednp as snp
import itertools

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
    check_baseline,
    eye3x4,
    _remove_correspondences_with_ids,
    Correspondences
)

range_of_neighbours = 75

triang_params = TriangulationParameters(
    max_reprojection_error=7.5,
    min_triangulation_angle_deg=1.0,
    min_depth=0.11)
iterations = 108
baseline_min_dist = 0

inliers_prob = 0.999
max_distance_to_epipolar_line = 1.5

homography_test_trashhold = 0.7

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
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = initialize_position(intrinsic_mat, corner_storage)

    

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
    print("Init -- ")
    print("\tTriangulated points -- ", len(pts3d))
    print("\tPoints in cloud -- ", len(point_cloud_builder.ids))
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
        print("Frame num -- ", best_for_pnp)
        print("\tInliers count -- ", len(inliers_for_best))
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
            print("\tTriangulated points -- ", len(pts3d))
        else:
            print("\tTriangulated points -- ", 0)
        print("\tPoints in cloud -- ", len(point_cloud_builder.ids))
        print()

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

def get_frames(frame_cnt):
    step = 20

    if frame_cnt < step:
        array_pairs = list(range(0, frame_cnt))
        return list(itertools.combinations(array_pairs, r=2))
    else:
        array_pairs = []
        for i in range(0, frame_cnt):
            for j in range(i, min(i + step, frame_cnt)):
                array_pairs.append((i, j))
        return array_pairs



def initialize_position(intrinsic_mat, corner_storage):
    np.random.seed(1337)
    min_corresp = 100
    good_treshold = 900

    frame_cnt = len(corner_storage)
    frames = get_frames(frame_cnt)
    best_view = None
    cnt = -1

    for known_ind1, known_ind2 in frames:
        frame1 = corner_storage[known_ind1]
        frame2 = corner_storage[known_ind2]
        
        correspondence = build_correspondences(frame1, frame2)
        if len(correspondence[0]) < min_corresp:
                continue
        E, inliers = cv2.findEssentialMat(
            correspondence.points_1,
            correspondence.points_2,
            intrinsic_mat,
            cv2.RANSAC,
            prob=0.999
        )
        correspondence = _remove_correspondences_with_ids(
                        correspondence, np.argwhere(inliers.flatten() == 0).astype(np.int64))

        homogr, homogr_inliers = cv2.findHomography(
            correspondence.points_1,
            correspondence.points_2,
            method=cv2.RANSAC
        )
        if np.count_nonzero(homogr_inliers) / np.count_nonzero(inliers) > homography_test_trashhold:
            continue 

        rot1, rot2, translation = cv2.decomposeEssentialMat(E)

        for rot in (rot1, rot2):
            for tran in (translation, -translation):
                view1 = eye3x4()
                view2 = np.hstack((rot, tran))

                _, pts, _ = triangulate_correspondences(correspondence, view1, view2, intrinsic_mat, triang_params)
                cnt_new = len(pts)
                if cnt_new > good_treshold:
                    return (known_ind1, view_mat3x4_to_pose(view1)), (known_ind2, view_mat3x4_to_pose(view2))
                if cnt < cnt_new:
                    best_view = view2
                    cnt = cnt_new

    
    return (known_ind1, view_mat3x4_to_pose(eye3x4())), (known_ind2, view_mat3x4_to_pose(best_view))

if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
