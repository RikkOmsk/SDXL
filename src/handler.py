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
    # pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe = StableDiffusionXLPipeline.from_single_file("https://civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com/model/2384906/socababesTurbo12.2zTA.safetensors?X-Amz-Expires=86400&response-content-disposition=attachment%3B%20filename%3D%22socababesTurboXL_v12Hybrid.safetensors%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=e01358d793ad6966166af8b3064953ad/20240420/us-east-1/s3/aws4_request&X-Amz-Date=20240420T083625Z&X-Amz-SignedHeaders=host&X-Amz-Signature=d2e781e5c9968edf658f9f70860aa98304e4f1448c4cb2c207e2984c8ef6e447", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", add_watermarker=False)
    pipe.to("cuda")
except RuntimeError:
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']

    time_start = time.time()
    image = pipe(prompt=prompt, height=1024, width=1024, num_inference_steps=10, guidance_scale=2.0).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    print(f"Time taken taken+buffer: {time.time() - time_start}")
    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
