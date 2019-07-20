import torch
import torchvision
import numpy as np
import runway

@runway.setup(options={})
def setup(opts):
    use_gpu = True if torch.cuda.is_available() else False
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celebAHQ-512', pretrained=True, useGPU=use_gpu)
    return model

@runway.command('generate',
                inputs={ 'z': runway.vector(512, sampling_std=0.5) },
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
    runway.run(host='0.0.0.0', port=8000)
