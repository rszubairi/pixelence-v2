import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D_Deep_Supervision_attention_cbam(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super(UNet3D_Deep_Supervision_attention_cbam, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        # Encoder path (3 layers)
        self.encoder1 = conv_block(self.in_channels, self.base_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = conv_block(self.base_filters, self.base_filters * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = conv_block(self.base_filters * 2, self.base_filters * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(self.base_filters * 4, self.base_filters * 8)

        # Decoder path (3 layers)
        self.up3 = nn.ConvTranspose3d(self.base_filters * 8, self.base_filters * 4, kernel_size=2, stride=2)
        self.decoder3 = conv_block(self.base_filters * 8, self.base_filters * 4)

        self.up2 = nn.ConvTranspose3d(self.base_filters * 4, self.base_filters * 2, kernel_size=2, stride=2)
        self.decoder2 = conv_block(self.base_filters * 4, self.base_filters * 2)

        self.up1 = nn.ConvTranspose3d(self.base_filters * 2, self.base_filters, kernel_size=2, stride=2)
        self.decoder1 = conv_block(self.base_filters * 2, self.base_filters)

        self.output_conv = nn.Conv3d(self.base_filters, self.out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        dec3 = self.decoder3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.up2(dec3)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))

        out = self.output_conv(dec1)
        return out

    def load_weights(self, weight_path, s3_bucket=None, s3_key=None):
        if s3_bucket and s3_key:
            from .s3_utils import download_from_s3
            from .decrypt_utils import decrypt_weights

            try:
                encrypted_weights_path = download_from_s3(s3_bucket, s3_key)
                decrypted_weights = decrypt_weights(encrypted_weights_path)
                self.load_state_dict(torch.load(decrypted_weights, map_location='cpu'))
                print("Weights loaded from S3.")
                return
            except Exception as e:
                print(f"Failed to load weights from S3: {e}")

        try:
            self.load_state_dict(torch.load(weight_path, map_location='cpu'))
            print("Weights loaded from local path.")
        except Exception as e:
            print(f"Failed to load weights from local path: {e}")
