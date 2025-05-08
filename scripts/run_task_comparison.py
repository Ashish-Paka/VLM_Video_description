#!/usr/bin/env python3
"""
Inference script for comparing robot and human videos from the RH20T dataset.
- Analyzes robot video with joint angle data
- Analyzes human video without joint angle data
- Compares the two videos to determine if they show the same task
"""

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
    transformed_dirs = list(task_path.glob("**/transformed"))
    if not transformed_dirs:
        raise FileNotFoundError(f"No transformed directory found in {task_path}")
    
    transformed_dir = transformed_dirs[0]
    joint_path = transformed_dir / "joint.npy"
    
    if not joint_path.exists():
        raise FileNotFoundError(f"Joint angles file not found at {joint_path}")
    
    # Load the joint angles
    joint_data = np.load(joint_path, allow_pickle=True).item()
    return joint_data


def generate_skill_description(video_b64: str, joint_data: dict, model: str = "models/gemini-1.5-pro") -> str:
    """
    Call Google Gemini multimodal API with RGB video and joint angles.
    Returns a concise description of the skill being performed.
    """
    # Create properly formatted media for the API
    prompt_message = (
        "Please analyze this skill demonstration carefully. "
        "I'm providing a video of a robotic task. "
        "Give me a detailed, step-by-step description of what's happening in this skill demonstration. "
        "Focus on: 1) The objects being manipulated, 2) The precise movements performed, "
        "3) The sequence of actions, and 4) The overall goal of the task."
    )

    try:
        genai_model = genai.GenerativeModel(model)
        video_part = {"mime_type": "video/mp4", "data": video_b64}
        response = genai_model.generate_content([prompt_message, video_part])
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"API Error: {str(e)}"


def find_video_path(base_path: Path, cam_id: str = "1"):
    """Find color video path for a given camera ID."""
    # Find camera directory matching cam_id
    cam_dirs = list(base_path.glob(f"**/*cam_{cam_id}*"))
    
    # Fallback: if specific cam_id not found, pick first camera directory available
    if not cam_dirs:
        cam_dirs = list(base_path.glob("**/*cam_*"))
        if not cam_dirs:
            raise FileNotFoundError(f"No camera directory found in {base_path}")
        print(f"Warning: camera {cam_id} not found, using {cam_dirs[0].name}")

    cam_dir = cam_dirs[0]
    
    # Look for color video
    color_paths = list(cam_dir.glob("*color*.mp4"))
    
    if not color_paths:
        raise FileNotFoundError(f"No RGB video found in {cam_dir}")
    
    return color_paths[0]


def compare_descriptions(robot_desc: str, human_desc: str, model: str = "gemini-1.5-flash") -> str:
    """
    Compare robot and human demonstration descriptions to categorize them as:
    1) Same skill
    2) Similar skills (with specific differences and similarities)
    3) Different skills
    """
    prompt_message = (
        "I have two descriptions of demonstrations - one from a robot and one from a human. "
        "Please analyze both descriptions and STRICTLY categorize them into ONE of these categories:\n"
        "1) SAME SKILL - The robot and human are performing fundamentally the same skill\n"
        "2) SIMILAR SKILLS - The robot and human are performing similar but not identical skills\n"
        "3) DIFFERENT SKILLS - The robot and human are performing completely different skills\n\n"
        "For your analysis:\n"
        "- Begin with a clear category label (SAME SKILL, SIMILAR SKILLS, or DIFFERENT SKILLS)\n"
        "- For SAME SKILL: Explain the core skill and why they are fundamentally the same despite any differences in execution\n"
        "- For SIMILAR SKILLS: List specific similarities, then specific differences\n"
        "- For DIFFERENT SKILLS: Explain why they are fundamentally different despite any surface similarities\n\n"
        f"Robot Demonstration:\n{robot_desc}\n\n"
        f"Human Demonstration:\n{human_desc}"
    )

    try:
        genai_model = genai.GenerativeModel(model)
        response = genai_model.generate_content(prompt_message)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini API for comparison: {str(e)}")
        return f"API Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemini vision inference on specific task videos for comparison"
    )
    parser.add_argument("--dataset_path", 
                        default="/home/ashish/Desktop/VLM/data/RH20T_cfg5",
                        help="Path to dataset folder")
    parser.add_argument("--task", 
                        default="task_0001_user_0010_scene_0001_cfg_0005", 
                        help="Task ID to process")
    parser.add_argument("--cam_id", default="1", help="Camera ID to use (default: 1)")
    parser.add_argument("--api_key", help="Google API key (or set GOOGLE_API_KEY env var)")
    args = parser.parse_args()

    # Configure Gemini API
    key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        parser.error("No API key provided. Use --api_key or set GOOGLE_API_KEY.")
    genai.configure(api_key=key)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        parser.error(f"Dataset path {dataset_path} doesn't exist")

    # Construct paths for robot and human demos
    robot_path = dataset_path / args.task
    human_path = dataset_path / f"{args.task}_human"

    if not robot_path.exists():
        parser.error(f"Robot demo path {robot_path} doesn't exist")
    if not human_path.exists():
        parser.error(f"Human demo path {human_path} doesn't exist")

    try:
        print(f"\nProcessing task: {args.task}")
        
        # Find videos
        print("\nLocating videos...")
        robot_video_path = find_video_path(robot_path, args.cam_id)
        human_video_path = find_video_path(human_path, args.cam_id)
        
        print(f"Robot video: {robot_video_path}")
        print(f"Human video: {human_video_path}")
        
        # Load joint angles (from robot demo only)
        print("\nLoading joint angles data for robot demonstration...")
        try:
            joint_data = load_joint_angles(robot_path)
            print("Joint data loaded successfully")
        except FileNotFoundError as e:
            print(f"Warning: {str(e)}")
            print("Proceeding without joint data")
            joint_data = {}
        
        # Load videos
        print("\nLoading video data...")
        robot_video_b64 = load_file_b64(str(robot_video_path))
        human_video_b64 = load_file_b64(str(human_video_path))
        
        # Run inference on robot video with joint data
        print("\nRunning robot video inference with Gemini (including joint data)...")
        robot_desc = generate_skill_description(robot_video_b64, joint_data)
        
        print("\nRobot skill description:")
        print("-" * 80)
        print(robot_desc)
        print("-" * 80)
        
        # Run inference on human video WITHOUT joint data
        print("\nRunning human demo inference with Gemini (no joint data)...")
        human_desc = generate_skill_description(human_video_b64, {})  # Empty dict instead of joint_data
        
        print("\nHuman demonstration description:")
        print("-" * 80)
        print(human_desc)
        print("-" * 80)
        
        # Compare the two descriptions
        print("\nComparing robot and human demonstrations...")
        comparison = compare_descriptions(robot_desc, human_desc)
        
        print("\nComparison analysis:")
        print("-" * 80)
        print(comparison)
        print("-" * 80)
        
        # Save results
        output_dir = Path(f"/home/ashish/Desktop/VLM/data/results/{args.task}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        robot_output = output_dir / f"robot_description_cam{args.cam_id}.txt"
        with open(robot_output, "w") as f:
            f.write(robot_desc)
        
        human_output = output_dir / f"human_description_cam{args.cam_id}.txt"
        with open(human_output, "w") as f:
            f.write(human_desc)
        
        comparison_output = output_dir / f"comparison_analysis_cam{args.cam_id}.txt"
        with open(comparison_output, "w") as f:
            f.write(comparison)
        
        print(f"\nResults saved to:")
        print(f"- Robot description: {robot_output}")
        print(f"- Human description: {human_output}")
        print(f"- Comparison analysis: {comparison_output}")
    
    except Exception as e:
        import traceback
        print(f"Error processing task: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())