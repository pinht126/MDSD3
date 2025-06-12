# Difference between Stable Diffusion v3
![image](https://github.com/user-attachments/assets/4b9f5aa1-2a7c-48bd-9214-56133ec9a4e6)

# Applying the Masking Algorithm to Stable Diffusion v3
![image](https://github.com/user-attachments/assets/93073c82-83e9-4433-811a-73c371598d10)

We apply the head-wise masking technique to Stable Diffusion v3, aiming to preserve its generation performance while enabling the creation of complex degraded images through the masking mechanism.

# Applying Clean Image Condition to Preserve Scene Image
![image](https://github.com/user-attachments/assets/b487f727-c67b-4647-8a6e-6d811479c22c)
We duplicate the part that receives the input image and insert clean image information by summing it with the output of a zero-convolution layer, allowing the model to preserve the original content.
However, we remove the modulation mechanism of AdaLN-Zero from the clean image input path, as the clean image information does not need to be influenced by the class conditioning.

# Results
![image](https://github.com/user-attachments/assets/2ecf9236-d1f8-40f9-8630-b7e0e8b55f7d)
Haze and rain both have the characteristic of increasing overall brightness. When both conditions are applied simultaneously, this can lead to the rain effect becoming overly blurred.

# Modify the masking ratio to a (float) value other than 1 and 0 when generating multi-degradation images.
![image](https://github.com/user-attachments/assets/686a6a2e-8088-4aec-bd52-5f7a60145e44)

![image](https://github.com/user-attachments/assets/7495555c-f969-45d9-9912-be83d2df2744)

# Adjust the Initial Noise using the noise equation!
![image](https://github.com/user-attachments/assets/7a8ef04f-c151-4620-9d9d-88359a80613f)
![image](https://github.com/user-attachments/assets/fdd6ae58-d30a-4981-89d3-4e330127f8bf)

noise equation : 𝑙𝑎𝑡𝑒𝑛𝑡𝑠=𝛼∗𝑛𝑜𝑖𝑠𝑒+𝛽∗𝑖𝑛𝑝𝑢𝑡 (𝛼+𝛽=1)


As noise decreases, the degradation condition becomes weaker.
Therefore, we want the sum of alpha and beta to be greater than 1.
Initail noise equation : 𝑙𝑎𝑡𝑒𝑛𝑡𝑠=𝛼∗𝑛𝑜𝑖𝑠𝑒+𝛽∗𝑖𝑛𝑝𝑢𝑡 (𝛼+𝛽>1)

![image](https://github.com/user-attachments/assets/de39e0b6-71bc-443d-ac9a-19d508991a56)

# Results

![image](https://github.com/user-attachments/assets/d26b2524-6ca9-43cf-9ad0-e176484e5a71)

![image](https://github.com/user-attachments/assets/66a96b51-e980-4ec1-8f64-9c018742c95f)

Problem: The generated results tend to preserve the overall color structure of the initial input image.

# Frequency-Domain Analysis

Step-wise generation results are presented for the haze, rain, and haze&rain classes.
![image](https://github.com/user-attachments/assets/1f28bd6d-90a9-41bf-afbb-c7897ca56f43)
 It is observed that haze, being a low-frequency degradation, is generated in the early steps of the diffusion model, whereas rain, which has high-frequency characteristics, is generated in the later steps.

 ![image](https://github.com/user-attachments/assets/6f2b2d72-091d-4a01-840e-cbc60b6db905)
[Boosting diffusion models with moving average sampling in frequency domain Qian et al, CVPR 2024]
Qian et al. stated that “Diffusion models at the denoising process first focus on the recovery of low-frequency components in the earlier timesteps and gradually shift to recovering high-frequency details in the later timesteps.”

-> So, Degradation-specific details(rain) should be generated in the later stages of the denoising process.

# Applying the focal-frequency loss to incorporate frequency-domain
![image](https://github.com/user-attachments/assets/2e9cc86b-ee5f-4403-bbfa-25cd36bb1acf)

[Jiang, Liming, et al. "Focal frequency loss for image reconstruction and synthesis." Proceedings of the IEEE/CVF international conference on computer vision. 2021.]
L jiang et al. use a frequency-domain loss instead of pixel-based loss when training GANs or VAEs  to better learn high-frequency details.


![image](https://github.com/user-attachments/assets/11a3a5f1-fedf-43c9-bbbc-03de496f9cbf)
We train the model to learn the degradation details(high-frequency).

Since frequency components become more important in the later stages of the backward process (i.e., at smaller timesteps), we multiply the focal-frequency loss by a weighting factor of (1 - T / 1000) to assign greater importance when T is small.

# results
![image](https://github.com/user-attachments/assets/2f78baf9-811b-4dac-8850-9710eb478845)
Artificial noise is suppressed, resulting in the effective generation of images containing a mixture of rain and haze degradations.
It shows visually effective results in specific style mixing scenarios.







## Installation

We recommend installing 🤗 Diffusers in a virtual environment from PyPI or Conda. For more details about installing [PyTorch](https://pytorch.org/get-started/locally/) and [Flax](https://flax.readthedocs.io/en/latest/#installation), please refer to their official documentation.

### PyTorch

With `pip` (official package):

```bash
pip install --upgrade diffusers[torch]
```

With `conda` (maintained by the community):

```sh
conda install -c conda-forge diffusers
```

### Flax

With `pip` (official package):

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) support

Please refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide.

## Quickstart

Generating outputs is super easy with 🤗 Diffusers. To generate an image from text, use the `from_pretrained` method to load any pretrained diffusion model (browse the [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) for 30,000+ checkpoints):

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

You can also dig into the models and schedulers toolbox to build your own diffusion system:

```python
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(50)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
image
```

Check out the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to launch your diffusion journey today!

# MDSD3
