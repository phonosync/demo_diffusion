import torch
from PIL import Image
import imageio
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline
    )

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("running on", device)

model_id = "stabilityai/stable-diffusion-2-1-base"
text2ImgPipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)
img2ImgPipeline = StableDiffusionImg2ImgPipeline.from_pipe(text2ImgPipeline).to(device)
inpaintPipeline = StableDiffusionInpaintPipeline.from_pipe(text2ImgPipeline).to(device)

def randomImage(num_inference_steps=20):
    return text2image(num_inference_steps=num_inference_steps, seed=0)

def text2image(prompt="",
               negative_prompt="Oversaturated, blurry, low quality",
               height=480, width=640,
               guidance_scale=8,
               num_inference_steps=20,
               seed=0, # 0 => random
               create_gif=False):
    generator = torch.Generator(device=device).manual_seed(seed)

    images = []
    # latents_sized_images = []

    def decode_tensors(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        # latents_sized_images.append(latents_to_rgb(latents))
        images.append(latents_to_pil(pipe.vae, latents))

        return callback_kwargs

    generated_image = text2ImgPipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height, width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator if seed != 0 else None,
        callback_on_step_end=decode_tensors if create_gif else None,
        callback_on_step_end_tensor_inputs=["latents"] if create_gif else None
        ).images[0]

    if create_gif:
        images.append(generated_image)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        gif_path = f"diffusion_{timestamp}.gif"
        imageio.mimsave(gif_path, np.array(images), fps=5, loop=0)
        # imageio.mimsave(f"diffusion_{timestamp}_latents_sized.gif", np.array(latents_sized_images), fps=3, loop=0)
        return generated_image, gif_path
    else:
        return generated_image

def text2image_latent_walk_image(prompt,
               negative_prompt="Oversaturated, blurry, low quality",
               height=480, width=640,
               guidance_scale=8,
               num_inference_steps=20,
               seed=0 # 0 => random
               ):
    generator = torch.Generator(device=device).manual_seed(seed)

    last_latents = []

    def save_last_latents(pipe, step, timestep, callback_kwargs):
        last_latents.append(callback_kwargs["latents"])
        return callback_kwargs

    generated_image = text2ImgPipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height, width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator if seed != 0 else None,
        callback_on_step_end=save_last_latents,
        callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

    images = []

    for _ in range(20):
        images.append(latents_to_pil(text2ImgPipeline.vae, last_latents[-1]+torch.rand(last_latents[-1].shape).to(device)))

    images.append(generated_image)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gif_path = f"diffusion_{timestamp}.gif"
    imageio.mimsave(gif_path, np.array(images), fps=5, loop=0)

    return generated_image, gif_path

def text2image_latent_walk_prompt(prompt,
               negative_prompt="Oversaturated, blurry, low quality",
               height=480, width=640,
               guidance_scale=8,
               num_inference_steps=20,
               seed=0 # 0 => random
               ):

    last_latents = []

    def save_last_latents(pipe, step, timestep, callback_kwargs):
        last_latents.append(callback_kwargs["latents"])
        return callback_kwargs
    
    prompt_embeddings = text2ImgPipeline.text_encoder(text2ImgPipeline.tokenizer(prompt, return_tensors='pt').to(device).input_ids)[0]

    images = []

    for _ in range(10):
        generator = torch.Generator(device=device).manual_seed(seed)
        generated_image = text2ImgPipeline(
            prompt_embeds=prompt_embeddings+torch.rand(prompt_embeddings.shape).to(device)*0.1,
            negative_prompt=negative_prompt,
            height=height, width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator if seed != 0 else None,
            callback_on_step_end=save_last_latents,
            callback_on_step_end_tensor_inputs=["latents"]
            ).images[0]
        images.append(generated_image)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gif_path = f"diffusion_{timestamp}.gif"
    imageio.mimsave(gif_path, np.array(images), fps=5, loop=0)

    return generated_image, gif_path

def text2image_prompt_merging(prompt1, prompt2, mix_factor=0.5,
               negative_prompt="Oversaturated, blurry, low quality",
               height=480, width=640,
               guidance_scale=8,
               num_inference_steps=20,
               seed=0 # 0 => random
               ):

    last_latents = []

    def save_last_latents(pipe, step, timestep, callback_kwargs):
        last_latents.append(callback_kwargs["latents"])
        return callback_kwargs
    
    prompt1_embeddings = text2ImgPipeline.text_encoder(text2ImgPipeline.tokenizer(prompt1, return_tensors='pt').to(device).input_ids)[0]
    prompt2_embeddings = text2ImgPipeline.text_encoder(text2ImgPipeline.tokenizer(prompt2, return_tensors='pt').to(device).input_ids)[0]
    prompt_embeddings = (prompt1_embeddings*mix_factor + prompt2_embeddings*(1-mix_factor))

    generator = torch.Generator(device=device).manual_seed(seed)
    generated_image = text2ImgPipeline(
        prompt_embeds=prompt_embeddings,
        negative_prompt=negative_prompt,
        height=height, width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator if seed != 0 else None,
        callback_on_step_end=save_last_latents,
        callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

    return generated_image

def text2image_prompt_interpolation(prompt1, prompt2,
               num_interpolation_steps=10,
               negative_prompt="Oversaturated, blurry, low quality",
               height=480, width=640,
               guidance_scale=8,
               num_inference_steps=20,
               seed=0 # 0 => random
               ):
    
    prompt1_embeddings = text2ImgPipeline.text_encoder(text2ImgPipeline.tokenizer(prompt1, return_tensors='pt').to(device).input_ids)[0]
    prompt2_embeddings = text2ImgPipeline.text_encoder(text2ImgPipeline.tokenizer(prompt2, return_tensors='pt').to(device).input_ids)[0]

    images = []

    for step in range(num_interpolation_steps+1):
        mix_factor = step/num_interpolation_steps
        prompt_embeddings = (prompt1_embeddings*mix_factor + prompt2_embeddings*(1-mix_factor))
        generator = torch.Generator(device=device).manual_seed(seed)
        generated_image = text2ImgPipeline(
            prompt_embeds=prompt_embeddings,
            negative_prompt=negative_prompt,
            height=height, width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator if seed != 0 else None
            ).images[0]
        images.append(generated_image)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gif_path = f"diffusion_{timestamp}.gif"
    imageio.mimsave(gif_path, np.array(images), fps=5, loop=0)

    return generated_image, gif_path

def image2image(prompt,
               image_path="",
               image=None,
               strength=0.4,
               negative_prompt="Oversaturated, blurry, low quality",
               height=480, width=640,
               guidance_scale=8,
               num_inference_steps=20,
               seed=0, # 0 => random
               create_gif=False):
    generator = torch.Generator(device=device).manual_seed(seed)

    images = []

    def decode_tensors(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        images.append(latents_to_pil(pipe.vae, latents))

        return callback_kwargs
    
    if image is None:
        image = Image.open(image_path)
    image.thumbnail((512, 512))

    generated_image = img2ImgPipeline(
        prompt=prompt,
        image=image,
        strength=strength,
        negative_prompt=negative_prompt,
        height=height, width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator if seed != 0 else None,
        callback_on_step_end=decode_tensors if create_gif else None,
        callback_on_step_end_tensor_inputs=["latents"] if create_gif else None,
        ).images[0]

    if create_gif:
        images.append(generated_image)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        gif_path = f"diffusion_{timestamp}.gif"
        imageio.mimsave(gif_path, np.array(images), fps=5, loop=0)

    return generated_image

def inpaint(prompt,
            image_path="",
            init_image=None,
            mask_path="",
            mask_image=None,
            negative_prompt="Oversaturated, blurry, low quality",
            height=480, width=640,
            guidance_scale=8,
            num_inference_steps=20,
            seed=0, # 0 => random
            create_gif=False):
    generator = torch.Generator(device=device).manual_seed(seed)

    images = []

    def decode_tensors(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        images.append(latents_to_pil(pipe.vae, latents))

        return callback_kwargs

    if init_image is None:
        init_image = Image.open(image_path).resize((512, 512))
    else:
        init_image = init_image.resize((512,512))

    if mask_image is None:
        mask_image = Image.open(mask_path).resize((512, 512))
    else:
        mask_image = mask_image.resize((512,512))
    
    generated_image = inpaintPipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        height=height, width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        
        generator=generator if seed != 0 else None,
        callback_on_step_end=decode_tensors if create_gif else None,
        callback_on_step_end_tensor_inputs=["latents"] if create_gif else None
        ).images[0]

    if create_gif:
        images.append(generated_image)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        gif_path = f"diffusion_{timestamp}.gif"
        imageio.mimsave(gif_path, np.array(images), fps=5, loop=0)

    return generated_image

def latents_to_pil(vae, latents):
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_image = Image.fromarray(images[0])
    return pil_image

def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35)
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)

    return Image.fromarray(image_array)

def plot_two_images(img1, img2):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, img in zip(axes, [img1, img2]):
        ax.imshow(img)
        ax.axis('off')
    plt.show()

def encode_image(image):
    image_bytes_io = BytesIO()
    image.save(image_bytes_io, format='PNG')
    formatted_image_bytes = image_bytes_io.getvalue()
    encoded = str(base64.b64encode(formatted_image_bytes), 'utf-8')
    return "data:image/png;base64,"+encoded