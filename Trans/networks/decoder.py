import torch
import torch.nn as nn



class ImageDecoder(nn.Module):
    """
    A PyTorch neural network module that decodes a latent tensor into an image.

    This decoder takes a latent tensor of shape [N, 768, 64, 64] and
    upsamples it to a final image of shape [N, 3, 512, 512].
    The architecture uses a series of transposed convolutions for upsampling,
    each followed by batch normalization and a ReLU activation function to
    progressively build the image from the latent representation.
    """
    def __init__(self, latent_dim=768, out_channels=3):
        """
        Initializes the ImageDecoder module.

        Args:
            latent_dim (int): The number of channels in the input latent tensor.
            out_channels (int): The number of channels in the output image (e.g., 3 for RGB).
        """
        super(ImageDecoder, self).__init__()

        # The decoder is built as a sequence of upsampling blocks.
        # We start with the latent dimension and progressively increase
        # the spatial dimensions while reducing the feature channels.
        self.decoder = nn.Sequential(
            # --- Block 1: Upsample from 64x64 to 128x128 ---
            # Input: [N, 768, 64, 64]
            # ConvTranspose2d is a learnable upsampling layer. With stride=2,
            # it doubles the height and width of the input tensor.
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Shape: [N, 512, 128, 128]

            # --- Block 2: Upsample from 128x128 to 256x256 ---
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Shape: [N, 256, 256, 256]

            # --- Block 3: Upsample from 256x256 to 512x512 ---
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Shape: [N, 128, 512, 512]

            # --- Final Output Layer ---
            # A final standard convolution maps the feature channels to the desired
            # number of output channels (3 for an RGB image) without changing size.
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            # Shape: [N, 3, 512, 512]

            # The Tanh activation function scales the output pixel values to the
            # range [-1, 1], which is a common convention for generated images.
            nn.Tanh()
        )

    def forward(self, x):
        """
        Defines the forward pass of the decoder.

        Args:
            x (torch.Tensor): The input latent tensor of shape [N, latent_dim, 64, 64].

        Returns:
            torch.Tensor: The output image tensor of shape [N, out_channels, 512, 512].
        """
        return self.decoder(x)


# class ImageDecoder(nn.Module):
#     """
#     A PyTorch neural network module that decodes a latent tensor into an image.

#     This decoder takes a latent tensor of shape [N, 768, 32, 32] and
#     upsamples it to a final image of shape [N, 3, 512, 512].
#     The architecture uses a series of transposed convolutions for upsampling,
#     each followed by batch normalization and a ReLU activation function to
#     progressively build the image from the latent representation.
#     """
#     def __init__(self, latent_dim=768, out_channels=3):
#         """
#         Initializes the ImageDecoder module.

#         Args:
#             latent_dim (int): The number of channels in the input latent tensor.
#             out_channels (int): The number of channels in the output image (e.g., 3 for RGB).
#         """
#         super(ImageDecoder, self).__init__()

#         # The decoder is built as a sequence of upsampling blocks.
#         # We start with the latent dimension and progressively increase
#         # the spatial dimensions while reducing the feature channels.
#         self.decoder = nn.Sequential(
#             # --- Block 1: Upsample from 32x32 to 64x64 ---
#             # Input: [N, 768, 32, 32]
#             # ConvTranspose2d is a learnable upsampling layer. With stride=2,
#             # it doubles the height and width of the input tensor.
#             nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             # Shape: [N, 512, 64, 64]

#             # --- Block 2: Upsample from 64x64 to 128x128 ---
#             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             # Shape: [N, 256, 128, 128]

#             # --- Block 3: Upsample from 128x128 to 256x256 ---
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             # Shape: [N, 128, 256, 256]

#             # --- Block 4: Upsample from 256x256 to 512x512 ---
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             # Shape: [N, 64, 512, 512]

#             # --- Final Output Layer ---
#             # A final standard convolution maps the feature channels to the desired
#             # number of output channels (3 for an RGB image) without changing size.
#             nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
#             # Shape: [N, 3, 512, 512]

#             # The Tanh activation function scales the output pixel values to the
#             # range [-1, 1], which is a common convention for generated images.
#             nn.Tanh()
#         )

#     def forward(self, x):
#         """
#         Defines the forward pass of the decoder.

#         Args:
#             x (torch.Tensor): The input latent tensor of shape [N, latent_dim, 32, 32].

#         Returns:
#             torch.Tensor: The output image tensor of shape [N, out_channels, 512, 512].
#         """
#         return self.decoder(x)



# --- Example Usage ---
# This block will only run when the script is executed directly.
if __name__ == '__main__':
    # Define parameters for the test
    batch_size = 4
    latent_channels = 768
    latent_size = 32
    output_channels = 3
    output_size = 512

    # Create a dummy input tensor to simulate the output of an encoder.
    # It has random values but the correct shape.
    dummy_latent = torch.randn(batch_size, latent_channels, latent_size, latent_size)
    print(f"Input latent shape: {dummy_latent.shape}")

    # Instantiate the decoder model
    decoder_model = ImageDecoder(latent_dim=latent_channels, out_channels=output_channels)
    # print(decoder_model) # Uncomment to see the detailed model architecture

    # Pass the dummy tensor through the decoder to get the output image
    try:
        output_image = decoder_model(dummy_latent)
        print(f"Output image shape: {output_image.shape}")

        # --- Verification ---
        expected_shape = (batch_size, output_channels, output_size, output_size)
        assert output_image.shape == expected_shape, \
            f"Shape mismatch! Expected {expected_shape}, but got {output_image.shape}"
        print("\nSuccessfully verified the output shape.")

    except Exception as e:
        print(f"\nAn error occurred during the forward pass: {e}")
