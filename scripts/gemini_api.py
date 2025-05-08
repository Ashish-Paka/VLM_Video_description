import google.generativeai as genai
from PIL import Image
import os
import argparse
import base64

# Set your API key as an environment variable before running the script
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Environment variable GOOGLE_API_KEY is not set. Please export your API key before running.")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Load the Gemini model (multimodal)
model = genai.GenerativeModel('models/gemini-1.5-pro')

# Add argument parsing to accept image or video
parser = argparse.ArgumentParser()
parser.add_argument("--media", required=True, help="Path to image or video file")
args = parser.parse_args()

# Load media based on file extension
media_path = args.media
if media_path.lower().endswith('.mp4'):
    with open(media_path, 'rb') as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode('utf-8')
    # SDK expects a Blob with 'mime_type' and 'data'
    media_input = {"mime_type": "video/mp4", "data": video_b64}
else:
    media_input = Image.open(media_path)

# Generate a response
response = model.generate_content(["What's in this media and explain what is happening with as tie passes?", media_input])

# Print the result
print(response.text)
