# 👁️ See-for-Me: Hands-Free Accessibility Assistant

**See-for-Me** is a completely hands-free Edge-AI Accessibility Assistant designed for the visually impaired. Unlike traditional apps that send sensitive personal photos to cloud servers, See-for-Me runs entirely locally on the user's device. 

By combining **Google PaliGemma** (Vision-Language Model) and **OpenAI Whisper** (Speech-to-Text), the app allows users to simply ask questions out loud about their surroundings. The AI analyzes the live camera feed and speaks the answer back to the user instantly. It's 100% private, requires zero cloud computing, and democratizes accessibility through Edge AI.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Gradio (Web UI, Webcam, Microphone)
* **Vision AI:** Google PaliGemma 3B (`paligemma-3b-mix-224`)
* **Speech-to-Text:** OpenAI Whisper (`whisper-tiny.en`)
* **Text-to-Speech:** Google TTS (`gTTS`)

## 🚀 How to Run Locally
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install torch transformers pillow gradio huggingface_hub gtts accelerate librosa soundfile
   ```
3. Open `app.py` and replace `"YOUR_HUGGINGFACE_TOKEN"` with your actual Hugging Face access token (Make sure you have accepted the PaliGemma license on Hugging Face).
4. Run the app:
   ```bash
   python app.py
   ```
5. Open the local web link, allow camera/microphone access, and speak to the AI!

## 🔒 Privacy First
This application runs model inference locally on your machine. Audio and image data are not sent to any cloud APIs for generation, ensuring absolute privacy for the user.
