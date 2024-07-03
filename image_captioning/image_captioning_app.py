import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

pretrained = "Salesforce/blip-image-captioning-base"
processor = AutoProcessor.from_pretrained(pretrained)
model = BlipForConditionalGeneration.from_pretrained(pretrained)

def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    text = "the image of"
    inputs = processor(images=raw_image, text=text, return_tensors = "pt") #return pt tensors
    outputs = model.generate(**inputs,max_length=50) #caption should be max 50 tokens in length
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

iface.launch()