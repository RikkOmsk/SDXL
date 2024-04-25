""" Example handler file. """

import runpod
import torch
import base64
import io
import os
import time
from diffusers import StableDiffusionXLPipeline
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/opt/creds.json"

try:
    pipe = StableDiffusionXLPipeline.from_single_file("model.safetensors", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", add_watermarker=False)
    pipe.to("cuda")
except RuntimeError:
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']

    time_start = time.time()
    generator = torch.Generator(device="cuda").manual_seed(job_input['seed'])
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator, height=job_input['height'], width=job_input['width'], num_inference_steps=job_input['num_inference_steps'], guidance_scale=job_input['guidance_scale']).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()


    # return 
    # jpg = base64.b64encode(image_bytes).decode('utf-8')
    filename = "pic.jpg"

    temp_location = '/tmp/' + filename          #here
    with open(temp_location, "wb") as f:        
        f.write(image_bytes)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket("face-swap-images")
    outputFile = "SDXL/" + job_input['userID'] + "/" + job_input['documentID'] + ".jpg"
    blob = bucket.blob(outputFile)
    blob.upload_from_filename(temp_location)

    return {
        "success": true,
        "imageUrl": blob.public_url
    }


runpod.serverless.start({"handler": handler})
