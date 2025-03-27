from .opt import BaseOptions
import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    def __init__(self, model_path):
        """Initialize the pix2pix class with default options."""
        self.opt = BaseOptions()
        self.opt.checkpoints_dir = model_path
        super().__init__(self.opt)  # Use super() for proper initialization
        
        # Only need visual names for inference
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        
        # Only need generator for inference
        self.model_names = ['G']
        
        # Define generator network
        self.netG = networks.define_G(
            self.opt.input_nc, 
            self.opt.output_nc, 
            self.opt.ngf, 
            self.opt.netG, 
            self.opt.norm,
            not self.opt.no_dropout, 
            self.opt.init_type, 
            self.opt.init_gain, 
            self.gpu_ids
        )
        # Load the saved model
        self.setup(self.opt)

    def forward(self, image):
        """Transform input image using Pix2Pix (B to A direction).
        
        Args:
            image: numpy array of shape (H, W, 3) in range [0, 255]
            
        Returns:
            transformed image as numpy array of shape (H, W, 3) in range [0, 255]
        """
        # Convert numpy image to tensor, normalize, and move to device
        # Since we're going B to A, the input image is B
        input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = (input_tensor - 0.5) * 2.0
        real_B = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            fake_A = self.netG(real_B)
            
        # Convert back to numpy array
        output_image = fake_A[0].cpu().permute(1, 2, 0).numpy()
        output_image =  ((output_image + 1.0) / 2.0 * 255.0).astype('uint8')
        
        return output_image

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
