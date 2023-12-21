from diffusers import StableDiffusionPipeline
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pipe_kwargs = {
	"tokenizer": None,
	"safety_checker": None,
	"feature_extractor": None,
	"requires_safety_checker": False,
}

model_id = "/data/huggingface/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"

# model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, **pipe_kwargs)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")
