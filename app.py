import gradio as gr
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, pipeline
import os
from gtts import gTTS
import tempfile
import traceback

# 1. Setup
# SECURITY NOTE: Never commit your real token to GitHub! 
# Put your token back here to run locally, but do not push it.
HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN" 
os.environ["HF_TOKEN"] = HF_TOKEN

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# 2. Load Models
print("Loading AI Brains...")
model_id = "google/paligemma-3b-mix-224"
processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch_dtype, token=HF_TOKEN
).to(device)
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=device)
print("All Models loaded successfully!")

# 3. Core AI Function (SAFE MODE)
def analyze_image(image, text_question, audio_question):
    try:
        # Check empty inputs
        if image is None:
            return "Please click the webcam video to capture an image first.", None
            
        # FIX 1: Convert webcam image to standard RGB to prevent shape crashes
        image = image.convert("RGB")
            
        question = ""
        if audio_question:
            question = transcriber(audio_question)["text"].strip()
        elif text_question:
            question = text_question
            
        if not question:
            return "Please type a question or use the microphone.", None

        # AI Inference
        inputs = processor(text=question, images=image, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
        
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if result.startswith(question):
            result = result[len(question):].strip()
            
        # Voice Output
        try:
            tts = gTTS(text=result, lang='en')
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            audio_path = temp_file.name
        except Exception:
            audio_path = None
            
        return f"🗣️ You asked: {question}\n\n🤖 AI Answer: {result}", audio_path
        
    except Exception as e:
        # FIX 2: Catch all errors and print them to the UI
        error_msg = f"⚠️ SYSTEM ERROR:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, None

# 4. User Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ See-for-Me: Hands-Free Accessibility Assistant")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="1. Camera View (CLICK TO SNAP PHOTO)", sources=["webcam", "upload"])
            gr.Markdown("### 2. Ask your question")
            audio_question = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak to the AI")
            text_question = gr.Textbox(label="⌨️ Or type it manually", placeholder="e.g., Is the door open?")
            submit_btn = gr.Button("Analyze Surroundings", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Transcript & Answer", lines=8)
            output_audio = gr.Audio(label="Voice Output", autoplay=True)
            
    submit_btn.click(
        fn=analyze_image, 
        inputs=[input_image, text_question, audio_question], 
        outputs=[output_text, output_audio]
    )

if __name__ == "__main__":
    demo.launch()
