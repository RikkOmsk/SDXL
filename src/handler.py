""" Example handler file. """

import runpod
from diffusers import StableDiffusionXLPipeline
import torch
import base64
import io
import time

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:
    pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
except RuntimeError:
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']

    time_start = time.time()
    image = pipe(prompt=prompt, height=1024, width=1024, num_inference_steps=20, guidance_scale=0.0).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})