import json
import os
import cv2
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

def save_json(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def create_blur_dict(data): 
    """
    Create dictionary with key being the video path and values
    being the frame number with each blur classification.
    """
    blur_dict_total = {} 
    for datarow in data:
        blur_dict = {}
        projects = datarow['projects']
        labels = projects[list(projects.keys())[0]]['labels']
        if len(labels): 
            frames = labels[0]['annotations']['frames']
            for frame_idx in frames:
                curr_frames = frames[frame_idx]
                for classification in curr_frames['classifications']:
                    if classification['name'] == 'blurriness':
                        if frame_idx not in blur_dict:
                            blur_dict[frame_idx] = int(classification['radio_answer']['name'])
                        else: print(frame_idx)
        blur_dict_total[datarow['data_row']['global_key']] = blur_dict
    return blur_dict_total

def save_frames(blur_dict):
    blurry_frames = 'blurry_frames'
    os.makedirs(blurry_frames, exist_ok=True)
    frame_count = 0
    new_blur_dict = {}
    for video_path, frames in blur_dict.items():
        cap = cv2.VideoCapture(video_path)
        for frame_idx, blur_score in frames.items():
            actual_idx = int(frame_idx)-1
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_idx)
            ret, frame = cap.read() 
            frame_file = os.path.join(blurry_frames, f'{video_path.split("/")[-1]}_{actual_idx}.jpg')
            # cv2.imwrite(frame_file, frame)
            new_blur_dict[frame_file] = {
                "frame": frame_count, 
                "blurriness_score": blur_score
            }
            frame_count += 1
    return new_blur_dict

def extract_frames(blur_dict):
    new_blur_dict = {}
    frame_count = 0
    for video_path, frames in blur_dict.items():
        cap = cv2.VideoCapture(video_path)
        for frame_idx, blur_score in frames.items():
            actual_idx = int(frame_idx)-1
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_idx)
            ret, frame = cap.read() 
            frame_file = os.path.join(f'{video_path.split("/")[-1]}_{actual_idx}.jpg')
            new_blur_dict[frame_file] = {
                "frame_index": frame_count, 
                "frame": frame,
                "blurriness_score": blur_score
            }
            frame_count += 1
    return new_blur_dict

def train_test_split_frames(blur_dict):
    frames = list(blur_dict.keys())
    labels = list([val['blurriness_score'] for key, val in blur_dict.items()])
    train_frames, val_frames, train_labels, val_labels = train_test_split(frames, labels, test_size=0.2, 
                                                                          stratify=labels, 
                                                                          random_state=42)
    train_dict = {frame: label for frame,label in zip(train_frames, train_labels)}
    val_dict = {frame: label for frame,label in zip(val_frames, val_labels)}
    train_base_dir = 'blurry_output/train'
    val_base_dir = 'blurry_output/val'
    os.makedirs(train_base_dir, exist_ok=True)
    os.makedirs(val_base_dir, exist_ok=True)
    classes = set(labels)
    for c in classes:
        os.makedirs(os.path.join(train_base_dir, str(c)), exist_ok=True)
        os.makedirs(os.path.join(val_base_dir, str(c)), exist_ok=True)
    save_image(train_dict, train_base_dir)
    save_image(val_dict, val_base_dir)
        
def save_image(frame_dict, base_dir):
    for frame, cls in frame_dict.items():
        img = Image.open(frame)
        save_path = os.path.join(base_dir, str(cls), frame.split("/")[-1])
        img.save(save_path)

def blur_preprocessing(json_path):
    data = save_json(json_path)
    blur_dict = create_blur_dict(data)
    new_blur_dict = save_frames(blur_dict)
    train_test_split_frames(new_blur_dict)
    return new_blur_dict
    
    
