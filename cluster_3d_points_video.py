import numpy as np
import pickle
import argparse
from tqdm import tqdm
import os
import pdb
import pyvista as pv
import time

def load_data(pkl_path):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Error: Input file not found at '{pkl_path}'")

    with open(pkl_path, 'rb') as f:
        frame_list = pickle.load(f)
    print(f"[Info] Loaded {len(frame_list)} frames from: {pkl_path}")
    return frame_list

def radius_based_clustering(points, radius, desc="Clustering"):
    if len(points) == 0:
        return np.array([], dtype=int)

    remaining_points = points.copy()
    representative_indices = []
    original_indices = np.arange(len(points))

    pbar = tqdm(total=len(points), desc=desc)
    while len(remaining_points) > 0:
        seed_idx_in_remaining = np.random.randint(0, len(remaining_points))
        seed_pt = remaining_points[seed_idx_in_remaining]

        original_seed_idx = original_indices[seed_idx_in_remaining]
        representative_indices.append(original_seed_idx)

        distances = np.linalg.norm(remaining_points - seed_pt, axis=1)
        mask_to_remove = distances < radius

        pbar.update(np.sum(mask_to_remove))

        remaining_points = remaining_points[~mask_to_remove]
        original_indices = original_indices[~mask_to_remove]

    pbar.close()
    return np.array(representative_indices)

def find_new_points(current_frame, prev_frame):
    current_pts = current_frame['xyz']
    current_colors = current_frame['rgb']
    prev_pts = prev_frame['xyz']

    prev_pts_view = prev_pts.view([('', prev_pts.dtype)] * prev_pts.shape[1])
    current_pts_view = current_pts.view([('', current_pts.dtype)] * current_pts.shape[1])

    is_new_mask = ~np.isin(current_pts_view, prev_pts_view, assume_unique=True)
    flat_mask = is_new_mask.flatten()

    pdb.set_trace()

    return current_pts[flat_mask], current_colors[flat_mask]


def main(args):
    frame_list = load_data(args.input_path)

    frame_0 = frame_list[0]
    points_frame_0 = frame_0['xyz']
    colors_frame_0 = frame_0['rgb']

    print(f"clustering with radius {args.cluster_radius}")
    key_indices = radius_based_clustering(points_frame_0, args.cluster_radius, desc="Clustering Frame 0")

    key_points_xyz = points_frame_0[key_indices]
    key_points_rgb = colors_frame_0[key_indices]  # Also get colors for key points

    final_representative_frames = [{'xyz': key_points_xyz, 'rgb': key_points_rgb}]

    previous_frame = frame_0
    point_cache = []  # store tuples of (xyz, rgb)

    for i in tqdm(range(1, len(frame_list)), desc="Processing Frames"):
        current_frame = frame_list[i]

        original_pcl = current_frame['xyz']
        original_rgb = current_frame['rgb']

        # pdb.set_trace()

        # newly_added_xyz, newly_added_rgb = find_new_points(current_frame, previous_frame)
        num_prev_frame = previous_frame['xyz'].shape[0]
        newly_added_xyz = current_frame['xyz'][num_prev_frame:]
        newly_added_rgb = current_frame['rgb'][num_prev_frame:]

        if len(newly_added_xyz) > 0:
            point_cache.append((newly_added_xyz, newly_added_rgb))

        num_cached_points = sum(len(p[0]) for p in point_cache)

        if num_cached_points > args.cache_threshold:
            all_cached_xyz = np.vstack([xyz for xyz, rgb in point_cache])
            all_cached_rgb = np.vstack([rgb for xyz, rgb in point_cache])

            genuinely_new_mask = np.ones(len(all_cached_xyz), dtype=bool)
            for j, point_c in enumerate(all_cached_xyz):
                min_dist = np.min(np.linalg.norm(key_points_xyz - point_c, axis=1))
                if min_dist <= args.norm_threshold:
                    genuinely_new_mask[j] = False

            # pdb.set_trace()

            genuinely_new_points = all_cached_xyz[genuinely_new_mask]
            genuinely_new_colors = all_cached_rgb[genuinely_new_mask]

            if len(genuinely_new_points) > 0:
                new_rep_indices = radius_based_clustering(
                    genuinely_new_points, args.cluster_radius, desc=f"Clustering new points at Frame {i}"
                )

                if len(new_rep_indices) > 0:
                    new_key_points_xyz = genuinely_new_points[new_rep_indices]
                    new_key_points_rgb = genuinely_new_colors[new_rep_indices]

                    # Add new representatives to the main set
                    key_points_xyz = np.vstack([key_points_xyz, new_key_points_xyz])
                    key_points_rgb = np.vstack([key_points_rgb, new_key_points_rgb])
                    print(f"Frame {i}: Added {len(new_key_points_xyz)} new representative points. Total is now {len(key_points_xyz)}.")

            point_cache = []

        # ============ Plot ============
        if i == 1:
            pl = pv.Plotter()

            pv_points = pv.PolyData(original_pcl)
            pv_points['colors'] = original_rgb

            pv_pbd_points = pv.PolyData(key_points_xyz)

            pdb.set_trace()

            pl.add_points(
                pv_points,
                style='points',
                scalars='colors',
                # scalars=original_rgb,
                rgb=True,
                render_points_as_spheres=False,
                point_size=10
            )
            pl.add_points(
                pv_pbd_points,
                style='points',
                color='#FF7A30',
                # scalars=original_rgb,
                # rgb=True,
                render_points_as_spheres=True,
                point_size=20
            )
            pl.camera_position = [(0.7947078067451492, 2.269401432933786, 5.384683407459932),
                                  (-0.11845957580527006, -0.33262403076562275, 1.3366592629450953),
                                  (-0.03155330752957934, -0.8375329894822626, 0.545474912633797)]
            # pl.show(interactive_update=False)
            pl.show(interactive_update=True)
            # pdb.set_trace()

        else:
            # if len(point_cache) > 0:
            # pv_points.points = original_pcl
            # pv_points['colors'] = original_rgb
            # print(pv_points.points.shape)

            # pv_pbd_points.points = key_points_xyz

            pl.add_points(
                newly_added_xyz,
                scalars=newly_added_rgb,
                style='points',
                rgb=True,
                render_points_as_spheres=False,
                point_size=10
            )

            pl.add_points(
                new_key_points_xyz,
                color='#386641',
                style='points',
                # rgb=True,
                render_points_as_spheres=True,
                point_size=20
            )

        time.sleep(0.2)

        # pdb.set_trace()

        # pdb.set_trace()

        final_representative_frames.append({'xyz': key_points_xyz, 'rgb': key_points_rgb})
        previous_frame = current_frame

    with open(args.output_path, 'wb') as f:
        pickle.dump(final_representative_frames, f)

    print(f"Saved {len(final_representative_frames)} frames to: {args.output_path}")
    print(f"Final number of representative points: {len(final_representative_frames[-1]['xyz'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligently downsample a point cloud sequence while retaining RGB color.")
    parser.add_argument("--input_path", type=str,
                        default="/home/fei/Documents/Dataset/StereoMIS_P3_1_tracking_full_with_projections_2/StereoMIS_P3_1_tracking_full/all_frames_gaussians.pkl", help="Path to the input .pkl file.")
    parser.add_argument("--output_path", type=str, default="representative_points_sequence_with_color.pkl", help="Path to save the output .pkl file.")
    parser.add_argument("--cluster_radius", type=float, default=0.2, help="Radius for clustering.")
    parser.add_argument("--cache_threshold", type=int, default=500, help="Number of new points to accumulate before triggering a check.")
    parser.add_argument("--norm_threshold", type=float, default=0.15, help="Distance threshold to identify a point as a new feature.")

    args = parser.parse_args()
    main(args)
