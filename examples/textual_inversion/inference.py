# from torch import autocast
from diffusers import StableDiffusionPipeline
import torch

# model_id = "./textual_inversion_damier-hirst/learned_model.bin"
# pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

prompt = "A <agnes-cecile> style wall art, trending on artstation"

# hyperparameters = {
#     "learning_rate": 5e-04,
#     "scale_lr": True,
#     "max_train_steps": 3000,
#     "train_batch_size": 1,
#     "gradient_accumulation_steps": 4,
#     "seed": 42,
#     "output_dir": "./textual_inversion_damier-hirst"
# }

# with torch.cuda.amp.autocast(True):
#     image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# image.save("damien-hirst-backpack.png")

pipe = StableDiffusionPipeline.from_pretrained(
    "./textual_inversion_agnes-cecile",
    torch_dtype=torch.float16,
).to("cuda")

with torch.cuda.amp.autocast(True):
    image = pipe(prompt, num_inference_steps=80, guidance_scale=7.5).images[0]

image.save("agnes-cecile-backpack-1.png")