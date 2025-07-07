# lesson_1
Первое домашнее задание

#@title Необходимые функции
!pip install jax jaxlib
!pip -q install diffusers
!pip -q install transformers scipy ftfy accelerate
!pip -q install "ipywidgets>=7,<8"
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from google.colab import output
output.enable_custom_widget_manager()

stableDiffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
stableDiffusion = stableDiffusion.to("cuda")


def createImagesStableDiffusion(prompt='', rows=2, cols=2, iteration=20):
  # Запускаем генерацию
  images =  stableDiffusion([prompt] * (rows*cols), num_inference_steps=iteration).images
  w, h = images[0].size
  grid = Image.new('RGB', size=(cols*w, rows*h))
  grid_w, grid_h = grid.size

  for i, img in enumerate(images):
      grid.paste(img, box=(i%cols*w, i//cols*h))
  display(grid)

#@title Внес изменения: Дуэйн Джонсон и корги, смешные очки, пляж, песок, тусовка, комикс.
createImagesStableDiffusion('Dwayne Johnson and his corgi, funny glasses, beach, sand, party, fun, comic book, 4k, –ar 16:9 ', 2, 2, 50)
