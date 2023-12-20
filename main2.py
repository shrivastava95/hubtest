import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch
import os

pipeline_path = 'sdxl_pipeline_quickloader.pt'

if os.path.exists(pipeline_path):
    base = torch.load(pipeline_path)
    base.to("cuda")
else:
    # Load the model with specified parameters
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        safety_checker=None, 
    )
    base.to("cuda")
    torch.save(base, pipeline_path)

def query(prompt, negative_prompt, steps):
    # Generate image based on the text prompt
    image = base(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps).images[0]
    return image

# Setup Gradio interface
demo = gr.Interface(
    fn=query, 
    inputs=["text", "text", gr.Slider(0, 100, 25)], 
    outputs="image"
)

if __name__ == "__main__":
    demo.launch(show_api=False)

# prompts = {
#     './toy dataset SD/toy dataset SD images/fruitbowltable/original caption.png'    : "A bowl of fruits on a table", # positive prompts
#     './toy dataset SD/toy dataset SD images/fruitbowltable/object exclusion.png'    : "A bowl of fruits", # exclusion
#     './toy dataset SD/toy dataset SD images/fruitbowltable/predicate negation.png'  : "A bowl of fruits not on a table", # object substitution
#     './toy dataset SD/toy dataset SD images/fruitbowltable/object substitution.png' : "A bowl of fruits on a sofa", # replacement
# }
