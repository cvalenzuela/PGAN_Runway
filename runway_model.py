## -----
# Sample code to port a model to Runway
# Check out the blog post to learn more
# https://medium.com/runwayml/porting-a-machine-learning-model-from-github-to-runway-in-5-minutes-555c5c9310af
#
#
# Runway Python SDK: https://sdk.runwayml.com/en/latest/
## -----

# Import our dependencies
import torch
import numpy as np
import runway # Be sure to install it first!: pip3 install runway-python

# Setup the model, initialize weights, set the configs of the model, etc.
# Every model will have a different set of configurations and requirements.
# Check https://sdk.runwayml.com/en/latest/runway_module.html to see a complete
# list of supported configs. The setup function should return the model ready to
# be used.
@runway.setup(options={"checkpoint": runway.category(description="Pretrained checkpoints to use.",
                                                choices=['celebAHQ-512', 'celebAHQ-256', 'celeba'],
                                                default='celebAHQ-512')})
def setup(opts):
    checkpoint = opts['checkpoint']
    use_gpu = True if torch.cuda.is_available() else False
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name=checkpoint, pretrained=True, useGPU=use_gpu)
    return model

# Every model needs to have at least one command. Every command allows to send
# inputs and process outputs. To see a complete list of supported inputs and
# outputs data types: https://sdk.runwayml.com/en/latest/data_types.html
@runway.command('generate',
                inputs={ 'z': runway.vector(length=512, sampling_std=0.5)},
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

# Run the model
if __name__ == '__main__':
    runway.run(port=5232)
