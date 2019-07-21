import torch
import torchvision
import numpy as np
import runway

checkpoint_description = "Pretrained checkpoints to use."

@runway.setup(options={"checkpoint": runway.category(description=checkpoint_description,
                                                choices=['celebAHQ-512', 'celebAHQ-256', 'DTD', 'celeba'],
                                                default='celebAHQ-512')})
def setup(opts):
    checkpoint = opts['checkpoint']
    use_gpu = True if torch.cuda.is_available() else False
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name=checkpoint, pretrained=True, useGPU=use_gpu)
    return model

@runway.command('generate',
                inputs={ 'z': runway.vector(512, sampling_std=0.5)},
                outputs={ 'image': runway.image })
def generate(model, inputs):
    z = inputs['z']
    latents = z.reshape((1, 512))
    latents = torch.from_numpy(latents)
    
    with torch.no_grad():
        generated_image = model.test(latents.float())
    generated_image = generated_image.clamp(min=-1, max=1)
    generated_image = ((generated_image + 1.0) * 255 / 2.0)
    return generated_image[0].permute(1, 2, 0).numpy().astype(np.uint8)

if __name__ == '__main__':
    runway.run(port=5232)
