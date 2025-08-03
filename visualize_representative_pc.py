import argparse
import pickle
import numpy as np
import os
import time
import open3d as o3d

def load_data(pkl_path):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Error: Input file not found at '{pkl_path}'")
        
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise TypeError("Expected the .pkl file to contain a list of dictionaries.")
        
    print(f"[Info] Loaded {len(data)} frames.")
    return data

def main(args):
    all_frames_data = load_data(args.input_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Representative Points Sequence (with Color)")
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = args.point_size

    pcd = o3d.geometry.PointCloud()

    first_frame_data = all_frames_data[0]
    
    points = first_frame_data['xyz']
    colors = first_frame_data['rgb']

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors) # Use the loaded colors
    
    vis.add_geometry(pcd)

    print("\n[Info] Starting live visualization...")
    print("Press 'q' or close the window to exit.")
    
    keep_running = True
    frame_index = 1
    
    while keep_running:
        if frame_index < len(all_frames_data):
            frame_data = all_frames_data[frame_index]
            
            current_points = frame_data['xyz']
            current_colors = frame_data['rgb']
            
            pcd.points = o3d.utility.Vector3dVector(current_points)
            pcd.colors = o3d.utility.Vector3dVector(current_colors)
            
            vis.update_geometry(pcd)
            
            frame_index += 1
        else:
            frame_index = 0

        keep_running = vis.poll_events()
        vis.update_renderer()
        
        time.sleep(1.0 / args.fps)
        
    vis.destroy_window()
    print("finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize an evolving sequence of representative points with color.")
    parser.add_argument(
        "--input_path", 
        type=str, 
        # Updated default filename
        default="representative_points_sequence_with_color.pkl",
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=15,
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=5.0,
    )
    args = parser.parse_args()

    main(args)