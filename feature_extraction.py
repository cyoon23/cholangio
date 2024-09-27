import os
import cv2
from tqdm import tqdm
import numpy as np
import re
from ConvAutoencoder import *
import torch
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import subprocess
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from ultralytics import YOLO


def extract_frames(frames_dir, video_path, every_n_frames=60):
    os.makedirs(frames_dir, exist_ok=True)
    crop_y, crop_x, crop_width, crop_height = 240, 660, 600, 600
    cap = cv2.VideoCapture(video_path) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = len(str(total_frames//every_n_frames))
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"select='not(mod(n\,{every_n_frames}))',crop={crop_width}:{crop_height}:{crop_x}:{crop_y}",
        '-vsync', 'vfr',
        os.path.join(frames_dir, f'frame_%0{n}d.jpg')
    ]

    # Run FFmpeg and parse progress
    process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
    process.wait() 

def pad_zeros(input_string, total_length):
    """Pad the input_string with leading zeros to fill the total_length."""
    return input_string.zfill(total_length)

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(0))
    return 0 

def compute_distances(features):
    # Compute Euclidean distances
    distances = []
    for i in range(0, len(features) - 1):  # Assuming consecutive images are pairs
        f1 = features[i].reshape(-1) 
        f2 = features[i+1].reshape(-1) 
        dist = euclidean(f1, f2)
        distances.append(dist)
    return distances

def find_shots(threshold, dataset, distances):
    # Label the frame pairs
    labels = np.array([1 if dist > threshold else 0 for dist in distances])

    # Find shot boundaries
    shot_boundaries = np.where(labels[:-1] != labels[1:])[0] 

    # Segment images based on boundaries
    shots = []
    start = 0
    for boundary in shot_boundaries:
        shots.append([x[0] for x in dataset.imgs[start:boundary]])  # Collect image paths
        start = boundary
    shots.append([x[0] for x in dataset.imgs[start:]])  # Last segment
    return shot_boundaries, shots

def max_shots(full_dataset, distances):
    thresholds = list(range(round(np.min(distances)), round(np.max(distances))))
    num_shots = [len(find_shots(i, full_dataset, distances)[0]) for i in thresholds]
    return thresholds[num_shots.index(max(num_shots))]

def preprocessing(bucket_name, every_n_frames=60):
    os.makedirs(f"{bucket_name}_frames", exist_ok=True)
    subdirs = os.listdir(bucket_name) 
    for subdir in subdirs:
        video_paths = os.listdir(f"{bucket_name}/{subdir}")
        os.makedirs(f"{bucket_name}_frames/{subdir}")
        os.makedirs(f"{bucket_name}_frames/{subdir}/frames")
        counter = 0
        for video_path in video_paths: 
            extract_frames(f"{bucket_name}_frames/{subdir}/{video_path}", f"{bucket_name}/{subdir}/{video_path}", every_n_frames)
            for frame in os.listdir(f"{bucket_name}_frames/{subdir}/{video_path}"):
                os.rename(f"{bucket_name}_frames/{subdir}/{video_path}/{frame}", f"{bucket_name}_frames/{subdir}/frames/frame_{str(counter).zfill(6)}.jpg")
                counter += 1 
            os.rmdir(f"{bucket_name}_frames/{subdir}/{video_path}")
            
def single_video_preprocessing(bucket_name, video_name, every_n_frames=60):
    os.makedirs(f"{bucket_name}_frames", exist_ok=True)
    os.makedirs(f"{bucket_name}_frames/{video_name}", exist_ok=True)
    os.makedirs(f"{bucket_name}_frames/{video_name}/frames", exist_ok=True)
    video_paths = os.listdir(f"{bucket_name}/{video_name}")
    counter = 0
    for video_path in video_paths:
        extract_frames(f"{bucket_name}_frames/{video_name}/{video_path}", f"{bucket_name}/{video_name}/{video_path}", every_n_frames)
        for frame in os.listdir(f"{bucket_name}_frames/{video_name}/{video_path}"):
            os.rename(f"{bucket_name}_frames/{video_name}/{video_path}/{frame}", f"{bucket_name}_frames/{video_name}/frames/frame_{pad_zeros(str(counter), 6)}.jpg")
            counter += 1 
        os.rmdir(f"{bucket_name}_frames/{video_name}/{video_path}")

def feature_extraction(root_dir, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load('model_8'))
    model.eval()
    features = []
    with torch.no_grad():
        for data in data_loader:
            images, _ = data
            images = images.to(device)
            feature = model.encoder(images)
            features.append(feature.cpu().numpy())
    return features

def make_dataset(root_dir):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a fixed size
        transforms.ToTensor()           # Convert images to PyTorch tensors
    ])
    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    return full_dataset

def optical_flow(frames_dir):
    motion_energies = []
    files = [os.path.join(frames_dir, file) for file in sorted(os.listdir(frames_dir)) if re.search(r'(\d+).jpg', file)]
    prev_frame = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    for idx in tqdm(range(1, len(files)), desc="Analyzing motion"):
        file = files[idx]
        curr_frame = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if curr_frame is None:
            continue

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute motion energy
        energy = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        motion_energy = np.sum(energy)
        motion_energies.append(motion_energy)

        # Update previous frame
        prev_frame = curr_frame 
        
    return motion_energies 

def keyframes_least_motion(shots, motion_energies, frame_paths):
    keyframe_indices = []
    max_digits = 6
    for i in range(len(shots)):
        shot = shots[i] 
        if len(shot) > 0:
            kf_indices = [extract_number(filename.split("/")[-1])-1 for filename in shot if extract_number(filename) < len(motion_energies)]    
            min_energy_idx = min(kf_indices, key=lambda idx: motion_energies[idx])
            keyframe_indices.append(frame_paths[min_energy_idx])
    return keyframe_indices

def blurriness_pruning(keyframe_paths):
    keyframe_indices = []
    blurry_classifier = YOLO('classify_blurry.pt')
    class_dict = blurry_classifier.names
    results = blurry_classifier(keyframe_paths)
    for i in range(len(keyframe_paths)):
        if int(class_dict[results[i].probs.top1]) < 75:  
            keyframe_indices.append(keyframe_paths[i])
    return keyframe_indices

def save_keyframes(save_dir, keyframe_indices, threshold):
    max_digits = 6
    keyframes = []
    os.makedirs(f"{save_dir}/{threshold}", exist_ok=True)  # Ensure the save directory exists
    for i in tqdm(range(len(keyframe_indices)), desc="Saving keyframes"):
        idx = keyframe_indices[i]
        frame_path = idx.split("/")[-1]
        img = cv2.imread(idx)
        cv2.imwrite(os.path.join(save_dir, str(threshold), frame_path), img)
        keyframes.append(frame_path)
    return keyframes 
    
def pipeline(bucket_name):
    preprocessing(bucket_name)
    frames_dir = f"{bucket_name}_frames"
    for root_dir in os.listdir(frames_dir):
        frames_subdir = f"{frames_dir}/{root_dir}"
        full_dataset = make_dataset(frames_subdir)
        features = feature_extraction(frames_subdir, full_dataset)
        distances = compute_distances(features)
        threshold = max_shots(full_dataset, distances)
        shot_boundaries, shots = find_shots(threshold, full_dataset, distances)
        motion_energies = optical_flow(f"{frames_subdir}/frames")
        frame_paths = [f"{frames_subdir}/frames/{frame_path}" for frame_path in sorted(os.listdir(f"{frames_dir}/frames"))]
        kf_least_motion = keyframes_least_motion(shots, motion_energies, frame_paths)
        pruned_keyframes = blurriness_pruning(kf_least_motion)
        save_keyframes(f"{root_dir}_keyframes", pruned_keyframes, threshold)
        
def single_video_pipeline(bucket_name, video_name):
    frames_dir = f"{bucket_name}_frames/{video_name}"
    single_video_preprocessing(bucket_name, video_name) 
    full_dataset = make_dataset(frames_dir)
    features = feature_extraction(frames_dir, full_dataset)
    distances = compute_distances(features)
    threshold = max_shots(full_dataset, distances)
    shot_boundaries, shots = find_shots(threshold, full_dataset, distances)
    motion_energies = optical_flow(f"{frames_dir}/frames") 
    frame_paths = [f"{frames_dir}/frames/{frame_path}" for frame_path in sorted(os.listdir(f"{frames_dir}/frames"))]
    kf_least_motion = keyframes_least_motion(shots, motion_energies, frame_paths)
    pruned_keyframes = blurriness_pruning(kf_least_motion)
    return save_keyframes(f"{video_name}_keyframes", pruned_keyframes, threshold)
       
def line_graph(arr_x, arr_y, title, x_label, y_label):
    plt.plot(arr_x, arr_y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
