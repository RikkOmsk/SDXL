""" Example handler file. """

import runpod
from diffusers import DPMSolverSinglestepScheduler, StableDiffusionXLPipeline
import torch
import base64
import io
import time


try:
    pipe = StableDiffusionXLPipeline.from_single_file("model.safetensors", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", add_watermarker=False)
    pipe.to("cuda")
except RuntimeError:
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']

    time_start = time.time()
    common_config = {'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': 'scaled_linear', 'use_karras_sigmas': True}
    scheduler = DPMSolverSinglestepScheduler(**common_config)
    pipe.scheduler = scheduler
    generator = torch.Generator(device="cuda").manual_seed(job_input['seed'])
    image = pipe(prompt=prompt, generator=generator, height=job_input['height'], width=job_input['width'], num_inference_steps=job_input['num_inference_steps'], guidance_scale=job_input['guidance_scale']).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    print(f"Time taken taken+buffer: {time.time() - time_start}")
    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
