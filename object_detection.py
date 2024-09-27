import json
import yaml
import os
import cv2
from preprocessing import save_json
import shutil
from sklearn.model_selection import train_test_split
import subprocess
import re
from ultralytics import YOLO
from tqdm import tqdm

def object_detection(json_path):
    data = save_json(json_path)
    frames_dir = 'object_detection_frames'
    labels_dir = 'object_detection_labels'
    output_dir = 'object_detection_output'
    yaml_dir = 'object_detection_dataset'
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    save_frames(data, frames_dir)
    class_map = save_labels(data, labels_dir)
    prepare_dataset(frames_dir, labels_dir, output_dir)
    make_dataset_yaml(yaml_dir, output_dir, class_map)
    model = YOLO("yolov8n.pt") 
    results = model.train(data=f"{yaml_dir}.yml", epochs=100, imgsz=640)
    best_model = YOLO(f"runs/detect/{most_recent_subdir('runs/detect', 'train')}/weights/best.pt")
    metrics = best_model.val() 
    
def save_frames(data, frames_dir): 
    existing_paths = set([file.split("_frame")[0] for file in os.listdir(frames_dir)])
    paths = list(set([datarow['data_row']['global_key'] for datarow in data]))
    for i in tqdm(range(len(paths))): 
        if paths[i].split("/")[-1] not in existing_paths:  
            extract_frames(paths[i], frames_dir, 1)   
    
def most_recent_subdir(directory, start_str):
    all_subdirs = [subdir for subdir in os.listdir(directory) if start_str in str(subdir)]
    return max(all_subdirs, key=os.path.getmtime)
    
def convert_to_yolo(bbox, img_width, img_height): 
    x_center = (bbox['left'] + bbox['width'] / 2) / img_width 
    y_center = (bbox['top'] + bbox['height'] / 2) / img_height
    width = bbox['width'] / img_width
    height = bbox['height'] / img_height 
    return float(x_center), float(y_center), float(width), float(height)

def extract_frames(video_path, frames_dir, every_n_frames=60):
    os.makedirs(frames_dir, exist_ok=True)
    video_name = video_path.split("/")[-1]
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"select='not(mod(n\,{every_n_frames}))'",
        '-vsync', 'vfr',
        os.path.join(f"{frames_dir}/{video_name}_frame_%04d.jpg")
    ]
    # Run FFmpeg and parse progress
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
    total = total_frames // every_n_frames
    process.wait()
    
def rename_files_in_directory(directory, video_name):
    """Rename files in the directory to have leading zeros for sorting."""
    
    # Determine the maximum number of digits needed
    filenames = [f for f in os.listdir(directory) if f.startswith('frame_') and f.endswith('.jpg')]

    for filename in filenames:
        # Create the new filename
        new_filename = f'{video_name}_{filename}'
        
        # Full path for old and new filenames
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_filepath, new_filepath)

def save_labels(data, output_dir):
    """
    Create dictionary with key being video path and values
    being frame index with each object type and bounding box
    {
        [video_path]: {
            [frame_index]: {
                [object_type]: {
                    top: top_dimension,
                    bottom: bottom_dimension,
                    left: left_dimension,
                    right: right_dimension
                }
            }
        }
    }
    """ 
    class_map = {}
    for datarow in data: 
        video_name = datarow['data_row']['global_key'].split("/")[-1]
        projects = datarow['projects']
        labels = projects[list(projects.keys())[0]]['labels']
        if len(labels): 
            frames = labels[0]['annotations']['frames']
            img_height = datarow['media_attributes']['height']
            img_width = datarow['media_attributes']['width']
            for frame_idx in frames:
                labels = []
                curr_frame = frames[frame_idx] 
                for obj in curr_frame['objects']:
                    curr_obj = curr_frame['objects'][obj]
                    bbox = curr_obj['bounding_box'] 
                    name = curr_obj['name'] 
                    if name not in class_map:
                        class_map[name] = len(class_map)
                    x_center, y_center, width, height = convert_to_yolo(bbox, img_height, img_width)
                    labels.append(f"{class_map[name]} {x_center} {y_center} {width} {height}")
                label_file = os.path.join(output_dir, f"{video_name}_frame_{int(frame_idx):04d}.txt")
                with open(label_file, 'w') as f:
                    f.write("\n".join(labels))
    return class_map
                    
def prepare_dataset(frames_dir, labels_dir, output_dir, test_size=0.2):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)

    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
    train_files, val_files = train_test_split(label_files, test_size=test_size)

    for f in train_files:
        shutil.copy(os.path.join(labels_dir, f), os.path.join(output_dir, 'train', 'labels', f))
        frames_file = f.replace('.txt', '.jpg')
        shutil.copy(os.path.join(frames_dir, frames_file), os.path.join(output_dir, 'train', 'images', frames_file))

    for f in val_files:
        shutil.copy(os.path.join(labels_dir, f), os.path.join(output_dir, 'val', 'labels', f))
        frames_file = f.replace('.txt', '.jpg')
        shutil.copy(os.path.join(frames_dir, frames_file), os.path.join(output_dir, 'val', 'images', frames_file))
        
def make_dataset_yaml(yaml_name, output_dir, class_map):
    train_path = f"../{output_dir}/train/images"
    val_path = f"../{output_dir}/val/images"
    names = [{v: k for k, v in class_map.items()}]
    d = [{
        "train": train_path,
        "val": val_path, 
        "names": names
    }]
    with open(f"{yaml_name}.yml", "w") as f:
        yaml.dump(d, f) 
