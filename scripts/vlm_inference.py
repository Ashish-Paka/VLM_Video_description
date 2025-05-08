#!/usr/bin/env python3
# vlm_inference.py - Run Gemini multimodal inference on robot and human demonstration videos
# Usage:
#   pip install google-generative-ai numpy
#   export GOOGLE_API_KEY="<your_api_key>"
#   python scripts/vlm_inference.py --dataset_path path/to/dataset --task_id task_id --cam_id cam_id

import os
import json
import base64
import argparse
from pathlib import Path
import numpy as np
import google.generativeai as genai


def load_file_b64(path: str) -> str:
    """Read binary file and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_joint_angles(task_path: Path) -> dict:
    """Load joint angles data from the task folder."""
    # Find the transformed folder in the task directory
    transformed_dir = list(task_path.glob("**/transformed"))[0]
    joint_path = transformed_dir / "joint.npy"
    
    if not joint_path.exists():
        raise FileNotFoundError(f"Joint angles file not found at {joint_path}")
    
    # Load the joint angles
    joint_data = np.load(joint_path, allow_pickle=True).item()
    return joint_data


def generate_skill_description(video_b64: str, joint_data: dict, depth_video_b64: str = None, model: str = "gemini-1.5-pro") -> str:
    """
    Call Google Gemini multimodal API with video, depth video, and joint angles.
    Returns a concise description of the skill being performed.
    """
    payload = {
        "rgb_video": {"content": video_b64, "mime_type": "video/mp4"},
        "joint_angles": joint_data
    }
    
    prompt_message = (
        "Please analyze this skill demonstration carefully. "
        "I'm providing an RGB video and joint angles sequence. "
    )
    
    if depth_video_b64:
        payload["depth_video"] = {"content": depth_video_b64, "mime_type": "video/mp4"}
        prompt_message = (
            "Please analyze this skill demonstration carefully. "
            "I'm providing BOTH RGB video AND depth video, along with joint angles sequence. "
            "The depth video provides additional spatial information. "
        )
    
    prompt_message += (
        "Give me a detailed, step-by-step description of what's happening in this skill demonstration. "
        "Focus on: 1) The objects being manipulated, 2) The precise movements performed, "
        "3) The sequence of actions, and 4) The overall goal of the task."
    )

    try:
        response = genai.chat.create(
            model=model,
            temperature=0.2,
            messages=[
                {"author": "user", "content": prompt_message},
                {"author": "user", "content": [
                    prompt_message,
                    {"type": "data", "value": json.dumps(payload)}
                ]}
            ]
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"API Error: {str(e)}"


def find_tasks(dataset_path: Path) -> list:
    """Find all task folders in the dataset."""
    # Look for task_* directories in the dataset
    task_dirs = list(dataset_path.glob("**/task_*"))
    return sorted(task_dirs)


def find_video_paths(base_path: Path, cam_id: str, is_human: bool = False):
    """Find color and depth video paths for a given camera ID."""
    # Adjust pattern based on if we're looking for human or robot demo
    if is_human:
        # For human demos, look for folders with "human" in the name
        human_paths = list(base_path.glob("*human*"))
        if not human_paths:
            raise FileNotFoundError(f"No human demonstration folder found in {base_path}")
        
        base_path = human_paths[0]
    
    # Find camera directory
    cam_dirs = list(base_path.glob(f"**/*cam_{cam_id}*"))
    if not cam_dirs:
        raise FileNotFoundError(f"No camera directory found for camera {cam_id}")
    
    cam_dir = cam_dirs[0]
    
    # Look for color and depth videos
    color_path = list(cam_dir.glob("*color*.mp4"))
    depth_path = list(cam_dir.glob("*depth*.mp4"))
    
    if not color_path:
        raise FileNotFoundError(f"No RGB video found in {cam_dir}")
    
    return color_path[0], depth_path[0] if depth_path else None


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemini VLM inference on robot and human demonstrations"
    )
    parser.add_argument("--dataset_path", required=True, help="Path to dataset folder")
    parser.add_argument("--task_id", help="Task ID to process (if not provided, will list available tasks)")
    parser.add_argument("--cam_id", default="1", help="Camera ID to use (default: 1)")
    parser.add_argument("--api_key", default=None, help="Google API key (or set GOOGLE_API_KEY env var)")
    args = parser.parse_args()

    # Configure Gemini API
    key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        parser.error("No API key provided. Use --api_key or set GOOGLE_API_KEY.")
    genai.configure(api_key=key)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        parser.error(f"Dataset path {dataset_path} doesn't exist")

    # Find tasks in the dataset
    tasks = find_tasks(dataset_path)
    if not tasks:
        parser.error(f"No task folders found in {dataset_path}")

    # If task_id not provided, list available tasks
    if not args.task_id:
        print("Available tasks:")
        for i, task_path in enumerate(tasks):
            print(f"  {i+1}. {task_path.name}")
        return

    # Find the specified task
    task_path = None
    for t in tasks:
        if args.task_id in t.name:
            task_path = t
            break

    if not task_path:
        parser.error(f"Task {args.task_id} not found in dataset")

    print(f"Processing task: {task_path.name}")

    try:
        # Process robot demonstration
        print("\nProcessing robot demonstration...")
        robot_color_path, robot_depth_path = find_video_paths(task_path, args.cam_id)
        print(f"Found robot videos: RGB={robot_color_path}, Depth={robot_depth_path}")

        # Process human demonstration
        print("\nProcessing human demonstration...")
        human_color_path, human_depth_path = find_video_paths(task_path, args.cam_id, is_human=True)
        print(f"Found human videos: RGB={human_color_path}, Depth={human_depth_path}")

        # Load joint angles data
        print("\nLoading joint angles data...")
        joint_data = load_joint_angles(task_path)
        print("Joint data loaded successfully")

        # Load videos
        print("\nLoading video data...")
        robot_color_b64 = load_file_b64(str(robot_color_path))
        robot_depth_b64 = load_file_b64(str(robot_depth_path)) if robot_depth_path else None
        
        human_color_b64 = load_file_b64(str(human_color_path))
        human_depth_b64 = load_file_b64(str(human_depth_path)) if human_depth_path else None

        # Generate descriptions
        print("\nRunning robot video inference with Gemini...")
        robot_desc = generate_skill_description(
            robot_color_b64, 
            joint_data, 
            depth_video_b64=robot_depth_b64
        )
        print("\nRobot skill description:")
        print("-" * 80)
        print(robot_desc)
        print("-" * 80)

        print("\nRunning human demo inference with Gemini...")
        human_desc = generate_skill_description(
            human_color_b64, 
            joint_data, 
            depth_video_b64=human_depth_b64
        )
        print("\nHuman demonstration description:")
        print("-" * 80)
        print(human_desc)
        print("-" * 80)
        
        # Save results
        output_dir = task_path / "vlm_results"
        output_dir.mkdir(exist_ok=True)
        
        robot_output = output_dir / f"robot_description_cam{args.cam_id}.txt"
        with open(robot_output, "w") as f:
            f.write(robot_desc)
        
        human_output = output_dir / f"human_description_cam{args.cam_id}.txt"
        with open(human_output, "w") as f:
            f.write(human_desc)
        
        print(f"\nResults saved to:")
        print(f"- Robot description: {robot_output}")
        print(f"- Human description: {human_output}")
        
    except Exception as e:
        import traceback
        print(f"Error processing task: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
