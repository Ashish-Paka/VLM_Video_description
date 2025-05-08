#!/usr/bin/env python3
"""
Deep Demonstration Description (ddd.py)
Analyzes video demonstrations of robot and human tasks from the RH20T dataset

This script:
- Processes both robot and human demonstration videos with timestamp metadata
- Applies different analysis prompts optimized for robotic vs human motion
- Generates detailed skill descriptions with trajectory paths, timestamps, and transforms
- Includes self-evaluated confidence ratings for each description
- Outputs separate entries per video into a structured CSV dataset

Usage:
  python scripts/ddd.py --dataset_path /home/ashish/Desktop/VLM/data/RH20T_cfg5 --output results.csv --parallel --workers 8
"""

import os
import argparse
import concurrent.futures
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from vlm_video_des import (
    load_file_b64,
    load_joint_angles,
    find_video_path
)


def generate_robot_skill_description(video_b64: str, joint_data: dict, model: str = "models/gemini-1.5-pro") -> tuple:
    """
    Generate a detailed analysis of robot skill demonstration with joint angle data
    
    Args:
        video_b64: Base64-encoded video data
        joint_data: Robot joint angles dictionary (if available)
        model: Gemini model name to use
    
    Returns:
        tuple: (detailed description text, self-confidence rating percentage)
    """
    prompt_message = (
        "You are a robotics expert analyzing a robot demonstration video. "
        "Please provide an extremely detailed analysis of this robotic demonstration, including:\n"
        "1. Objects being manipulated (properties, positions, orientations)\n"
        "2. Precise movements (trajectory paths, velocities, accelerations)\n"
        "3. Chronological sequence with approximate timestamps\n"
        "4. Robot end-effector state changes and interactions\n"
        "5. Overall goal and strategy of the task\n\n"
        "After your analysis, on a separate line, provide a single percentage value representing your confidence "
        "in the accuracy and completeness of your description (e.g., 'Confidence Rating: 87%'). "
        "Consider factors like video clarity, motion complexity, occlusions, etc."
    )
    
    try:
        # Initialize Gemini model and encode video for API
        genai_model = genai.GenerativeModel(model)
        video_part = {"mime_type": "video/mp4", "data": video_b64}
        
        # Call Gemini API for detailed analysis
        response = genai_model.generate_content([prompt_message, video_part])
        text = response.text.strip()
        
        # Extract self-confidence rating
        confidence = "0%"
        if "confidence rating:" in text.lower():
            parts = text.split("Confidence Rating:", 1)
            if len(parts) > 1:
                rating_part = parts[1].strip()
                percentage = rating_part.split("%")[0].strip() + "%"
                confidence = percentage
                # Remove the rating from the description text
                text = parts[0].strip()
        
        return text, confidence
    except Exception as e:
        print(f"Error calling Gemini API for robot description: {str(e)}")
        return f"API Error: {str(e)}", "0%"


def generate_human_skill_description(video_b64: str, model: str = "models/gemini-1.5-pro") -> tuple:
    """
    Generate a detailed analysis of human skill demonstration
    
    Args:
        video_b64: Base64-encoded video data
        model: Gemini model name to use
    
    Returns:
        tuple: (detailed description text, self-confidence rating percentage)
    """
    prompt_message = (
        "You are a human motion analysis expert reviewing a demonstration video. "
        "Please provide an extremely detailed analysis of this human demonstration video, including:\n"
        "1. Objects being manipulated (properties, positions, orientations)\n"
        "2. Precise movements and gestures (motion paths, timing, techniques)\n"
        "3. Chronological sequence with approximate timestamps\n"
        "4. Hand state changes and interactions with objects\n"
        "5. Overall goal and strategy of the demonstrated skill\n\n"
        "After your analysis, on a separate line, provide a single percentage value representing your confidence "
        "in the accuracy and completeness of your description (e.g., 'Confidence Rating: 87%'). "
        "Consider factors like video clarity, motion complexity, occlusions, etc."
    )
    
    try:
        # Initialize Gemini model and encode video for API
        genai_model = genai.GenerativeModel(model)
        video_part = {"mime_type": "video/mp4", "data": video_b64}
        
        # Call Gemini API for detailed analysis
        response = genai_model.generate_content([prompt_message, video_part])
        text = response.text.strip()
        
        # Extract self-confidence rating
        confidence = "0%"
        if "confidence rating:" in text.lower():
            parts = text.split("Confidence Rating:", 1)
            if len(parts) > 1:
                rating_part = parts[1].strip()
                percentage = rating_part.split("%")[0].strip() + "%"
                confidence = percentage
                # Remove the rating from the description text
                text = parts[0].strip()
        
        return text, confidence
    except Exception as e:
        print(f"Error calling Gemini API for human description: {str(e)}")
        return f"API Error: {str(e)}", "0%"


def main():
    parser = argparse.ArgumentParser(
        description="Deep Demonstration Description generator for robot and human videos"
    )
    parser.add_argument("--dataset_path", default="/home/ashish/Desktop/VLM/data/RH20T_cfg5",
                        help="Path to RH20T_cfg5 dataset root")
    parser.add_argument("--output", default="/home/ashish/Desktop/VLM/data/results/skill_descriptions.csv",
                        help="Path to output CSV file")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--subset", type=float, default=0.1, 
                        help="Process subset of data (0.1 = 10%, 1.0 = all)")
    args = parser.parse_args()

    # Configure Gemini API with fixed API key
    genai.configure(api_key="AIzaSyC1FCSddS98OGGjyR6MD9w_JvgCegKd1gU")

    # Verify dataset path exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        parser.error(f"Dataset path {dataset_path} doesn't exist")

    # Identify task folders that have matching human demonstration folders
    tasks = [d.name for d in dataset_path.glob("task_*")
             if d.is_dir() and not d.name.endswith("_human")
             and (dataset_path / f"{d.name}_human").exists()]
    
    total_tasks = len(tasks)
    if args.subset < 1.0:
        # Take a subset of tasks based on the subset parameter
        subset_count = max(1, int(args.subset * total_tasks))
        tasks = tasks[:subset_count]
        print(f"Processing {args.subset*100:.1f}% subset: {len(tasks)}/{total_tasks} tasks")
    else:
        print(f"Processing all {total_tasks} tasks")

    # Results container
    rows = []

    def process_one(task_id):
        """Process a single task folder and its human counterpart"""
        local_rows = []
        robot_path = dataset_path / task_id
        human_path = dataset_path / f"{task_id}_human"
        
        # Load joint angle data for robot demonstrations (once per task)
        joint_data = {}
        try:
            joint_data = load_joint_angles(robot_path)
        except Exception as e:
            print(f"Warning: Could not load joint data for {task_id}: {e}")
        
        # Process each camera directory in the task folder
        for cam_dir in robot_path.glob("cam_*"):
            if not cam_dir.is_dir():
                continue
                
            # Find matching human camera directory
            human_cam = human_path / cam_dir.name
            if not human_cam.exists():
                continue
                
            # Find RGB video files in both directories
            robot_vid = next(cam_dir.glob("*color*.mp4"), None)
            human_vid = next(human_cam.glob("*color*.mp4"), None)
            
            # Skip if neither video exists
            if not robot_vid and not human_vid:
                continue

            # Parse task identifiers from folder name
            parts = task_id.split("_")
            task_num = parts[1] if len(parts) > 1 else ""
            user_num = parts[3] if len(parts) > 3 else ""
            scene_num = parts[5] if len(parts) > 5 else ""
            cam_num = cam_dir.name  # Using full camera ID with timestamp

            # Process robot video with joint data if available
            if robot_vid:
                # Load video data
                rv_b64 = load_file_b64(str(robot_vid))
                
                # Generate robot skill description with joint angles
                robot_desc, robot_confidence = generate_robot_skill_description(rv_b64, joint_data)
                
                # Add row for robot video
                local_rows.append({
                    "Serial #": 0,  # Will be assigned sequentially later
                    "Robot folder": robot_path.name,
                    "Human folder": human_path.name,
                    "Task #": task_num,
                    "user #": user_num,
                    "scene #": scene_num,
                    "camera #": cam_num,
                    "robot skill description": robot_desc,
                    "human skill description": "",
                    "Description Rating percentage (self rated)": robot_confidence
                })
                tqdm.write(f"{task_id}/{cam_dir.name} robot done [Rating: {robot_confidence}]")

            # Process human video (without joint data)
            if human_vid:
                # Load video data
                hv_b64 = load_file_b64(str(human_vid))
                
                # Generate human skill description 
                human_desc, human_confidence = generate_human_skill_description(hv_b64)
                
                # Add row for human video
                local_rows.append({
                    "Serial #": 0,  # Will be assigned sequentially later
                    "Robot folder": robot_path.name,
                    "Human folder": human_path.name,
                    "Task #": task_num,
                    "user #": user_num,
                    "scene #": scene_num,
                    "camera #": cam_num,
                    "robot skill description": "",
                    "human skill description": human_desc,
                    "Description Rating percentage (self rated)": human_confidence
                })
                tqdm.write(f"{task_id}/{cam_dir.name} human done [Rating: {human_confidence}]")
                
        return local_rows

    # Process tasks in parallel if requested
    if args.parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, t): t for t in tasks}
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(tasks), desc="Processing tasks", unit="task"):
                task_rows = future.result()
                rows.extend(task_rows)
    else:
        # Process tasks sequentially
        for task_id in tqdm(tasks, desc="Processing tasks", unit="task"):
            task_rows = process_one(task_id)
            rows.extend(task_rows)

    # Assign sequential serial numbers to all rows
    for idx, row in enumerate(rows, start=1):
        row["Serial #"] = idx

    # Write results to CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"Results written to {args.output}")
        print(f"Processed {len(rows)} video descriptions")
        print(f"- Robot videos: {sum(1 for r in rows if r['robot skill description'])}")
        print(f"- Human videos: {sum(1 for r in rows if r['human skill description'])}")
    else:
        print("No task/video pairs found to process.")


if __name__ == "__main__":
    main()