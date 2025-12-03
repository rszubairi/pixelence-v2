import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced MRI Generator")

st.title("Enhanced MRI Generator")

# Device
device = 'cpu'

# Model size
size = (182, 182, 22)

# Model definition (copied from notebook)
class ContrastEnhancementBlock3D(nn.Module):
    def __init__(self, channels):
        super(ContrastEnhancementBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // 4, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        local = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.conv2, self.bn2
        )(x)
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attention = self.sigmoid(avg_out + max_out)
        out = x + local * attention
        return self.relu(out)

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)
        attn = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        return x * attn.expand_as(x)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn

class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction=8, spatial_kernel=7):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

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

        self.encoder1 = conv_block(self.in_channels, self.base_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = conv_block(self.base_filters, self.base_filters * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc_attent2 = CBAM3D(self.base_filters * 2)
        self.encoder3 = conv_block(self.base_filters * 2, self.base_filters * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc_attent3 = CBAM3D(self.base_filters * 4)
        self.bottleneck = conv_block(self.base_filters * 4, self.base_filters * 8)
        self.bottleneck_attent = CBAM3D(self.base_filters * 8)
        self.up3 = nn.ConvTranspose3d(self.base_filters * 8, self.base_filters * 4, kernel_size=2, stride=2)
        self.attent3 = CBAM3D(self.base_filters * 4)
        self.decoder3 = conv_block(self.base_filters * 8, self.base_filters * 4)
        self.dec3_drop = nn.Dropout3d(0.25)
        self.ds3_out = nn.Conv3d(self.base_filters * 4, self.out_channels, kernel_size=1)
        self.up2 = nn.ConvTranspose3d(self.base_filters * 4, self.base_filters * 2, kernel_size=2, stride=2)
        self.attent2 = CBAM3D(self.base_filters * 2)
        self.decoder2 = conv_block(self.base_filters * 4, self.base_filters * 2)
        self.dec2_drop = nn.Dropout3d(0.25)
        self.ds2_out = nn.Conv3d(self.base_filters * 2, self.out_channels, kernel_size=1)
        self.up1 = nn.ConvTranspose3d(self.base_filters * 2, self.base_filters, kernel_size=2, stride=2)
        self.attent1 = CBAM3D(self.base_filters)
        self.decoder1 = conv_block(self.base_filters * 2, self.base_filters)
        self.dec1_drop = nn.Dropout3d(0.25)
        self.contrast_enhance_block = ContrastEnhancementBlock3D(self.base_filters)
        self.output_conv = nn.Conv3d(self.base_filters, self.out_channels, kernel_size=1)

    def center_crop(self, tensor, target_shape):
        _, _, h, w, d = tensor.shape
        th, tw, td = target_shape
        h1 = (h - th) // 2
        w1 = (w - tw) // 2
        d1 = (d - td) // 2
        return tensor[:, :, h1:h1+th, w1:w1+tw, d1:d1+td]

    def pad_to_multiple(self, x, multiple=16):
        d, h, w = x.shape[2:]
        pad_d = (multiple - d % multiple) % multiple
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        pad = [pad_w // 2, pad_w - pad_w // 2,
               pad_h // 2, pad_h - pad_h // 2,
               pad_d // 2, pad_d - pad_d // 2]
        x_padded = F.pad(x, pad, mode='constant', value=0)
        return x_padded, pad

    def forward(self, x):
        original_shape = x.shape[2:]
        x, pad_sizes = self.pad_to_multiple(x, multiple=8)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.enc_attent2(enc2)
        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.enc_attent3(enc3)
        bottleneck = self.bottleneck(self.pool3(enc3))
        bottleneck = self.bottleneck_attent(bottleneck)
        dec3 = self.up3(bottleneck)
        enc3_cropped = self.center_crop(enc3, dec3.shape[2:])
        dec3 = self.decoder3(torch.cat([dec3, self.attent3(enc3_cropped)], dim=1))
        dec3 = self.dec3_drop(dec3)
        dec2 = self.up2(dec3)
        enc2_cropped = self.center_crop(enc2, dec2.shape[2:])
        dec2 = self.decoder2(torch.cat([dec2, self.attent2(enc2_cropped)], dim=1))
        dec2 = self.dec2_drop(dec2)
        dec1 = self.up1(dec2)
        enc1_cropped = self.center_crop(enc1, dec1.shape[2:])
        dec1 = self.decoder1(torch.cat([dec1, self.attent1(enc1_cropped)], dim=1))
        dec1 = self.dec1_drop(dec1)
        out = self.contrast_enhance_block(dec1)
        out = self.output_conv(out)
        ds2 = F.interpolate(self.ds2_out(dec2), size=out.shape[2:], mode='trilinear', align_corners=False)
        ds3 = F.interpolate(self.ds3_out(dec3), size=out.shape[2:], mode='trilinear', align_corners=False)
        out = out + 0.3 * ds2 + 0.2 * ds3
        out = self.center_crop(out, original_shape)
        return out

# Transform class
class Transform3D:
    def __init__(self, size=(240, 240, 144)):
        self.size = size

    def apply(self, sample):
        sample = self.contrast_normalize_fn(sample)
        sample = torch.from_numpy(sample)
        sample = self.center_crop_or_pad(sample, self.size)
        sample[-1] = self.float_normalize_fn(sample[:1])
        sample[:3] = self.float_normalize_fn(sample[:3])
        return sample

    def center_crop_or_pad(self, tensor, target_size):
        c, h, w, d = tensor.shape
        th, tw, td = target_size
        pad = [0, 0, 0, 0, 0, 0]
        crop = [0, 0, 0]
        for i, (cur, tgt) in enumerate(zip((h, w, d), target_size)):
            delta = tgt - cur
            if delta > 0:
                pad[2*i+1] = delta // 2
                pad[2*i] = delta - pad[2*i+1]
            else:
                crop[i] = -delta
        if any(p > 0 for p in pad):
            tensor = F.pad(tensor, pad[::-1], mode='constant', value=0)
        if any(c > 0 for c in crop):
            h_start = (tensor.shape[1] - th) // 2
            w_start = (tensor.shape[2] - tw) // 2
            d_start = (tensor.shape[3] - td) // 2
            tensor = tensor[:, h_start:h_start+th, w_start:w_start+tw, d_start:d_start+td]
        return tensor

    def float_normalize_fn(self, volume):
        return volume / (volume.max() + 1e-8)

    def contrast_normalize_fn(self, volume, lower_percentile=1, upper_percentile=99, blend_ratio=0.25):
        volume = volume.astype(np.float32)
        norm_volume = np.zeros_like(volume)
        C = volume.shape[0]
        for c in range(C):
            v = volume[c]
            v_flat = v.flatten()
            p_low = np.percentile(v_flat, lower_percentile)
            p_high = np.percentile(v_flat, upper_percentile)
            if p_high - p_low > 1e-5:
                v_stretch = np.clip((v - p_low) / (p_high - p_low), 0, 1)
            else:
                v_stretch = np.zeros_like(v)
            v_minmax = (v - v.min()) / (v.max() - v.min() + 1e-5)
            norm_volume[c] = (1 - blend_ratio) * v_minmax + blend_ratio * v_stretch
        return norm_volume

# Sharpen function
def sharpen_volume(volume: np.ndarray, background_threshold=0.01) -> np.ndarray:
    C, H, W, D = volume.shape
    sharpened = np.zeros_like(volume, dtype=np.float32)
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    for c in range(C):
        for d in range(D):
            slice_2d = volume[c, :, :, d]
            sharpened_slice = cv2.filter2D(slice_2d, -1, kernel)
            sharpened_slice = np.clip(sharpened_slice, 0.0, 1.0)
            background_mask = slice_2d < background_threshold
            sharpened_slice[background_mask] = slice_2d[background_mask]
            sharpened[c, :, :, d] = sharpened_slice
    return sharpened

# Load model
@st.cache_resource
def load_model():
    model = UNet3D_Deep_Supervision_attention_cbam(in_channels=3, out_channels=1, base_filters=32).to(device)
    model.load_state_dict(torch.load('../_global1.pth', map_location=device))
    model.eval()
    return model

model = load_model()
transform_ = Transform3D(size)

# Samples
samples_dir = '../_samples_'
sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.npy')]

st.sidebar.header("Select Sample")
selected_sample = st.sidebar.selectbox("Choose a patient sample", sample_files)

if selected_sample:
    sample_path = os.path.join(samples_dir, selected_sample)
    all_np = np.load(sample_path)[..., ::4]  # Downsample depth

    # Process input
    input_tensor = torch.cat([transform_.apply(all_np[i:i+1].copy()) for i in range(3)], dim=0)
    real_contrast_tensor = transform_.apply(all_np[-1:].copy())
    real_contrast = real_contrast_tensor.numpy()

    # Generate synthetic
    with torch.no_grad():
        model_input = input_tensor.unsqueeze(0).to(device)
        model_output = model(model_input)[0].cpu().numpy()
        model_output = np.clip(model_output, 0, None)
        model_output = (model_output - model_output.min()) / (model_output.max() - model_output.min() + 1e-8)
        synthetic = sharpen_volume(model_output)

    # Compute metrics (average over slices)
    ssim_values = []
    psnr_values = []
    for d in range(real_contrast.shape[3]):
        real_slice = real_contrast[0, :, :, d]
        syn_slice = synthetic[0, :, :, d]
        ssim_val = ssim(real_slice, syn_slice, data_range=1.0)
        psnr_val = psnr(real_slice, syn_slice, data_range=1.0)
        ssim_values.append(ssim_val)
        psnr_values.append(psnr_val)
    avg_ssim = np.mean(ssim_values)
    avg_psnr = np.mean(psnr_values)

    # Slice selector
    slice_idx = st.slider("Select Slice", 0, real_contrast.shape[3] - 1, real_contrast.shape[3] // 2)

    # Compute metrics for selected slice
    real_slice = real_contrast[0, :, :, slice_idx]
    syn_slice = synthetic[0, :, :, slice_idx]
    slice_ssim = ssim(real_slice, syn_slice, data_range=1.0)
    slice_psnr = psnr(real_slice, syn_slice, data_range=1.0)

    # Display
    st.header(f"Results for {selected_sample}")
    st.write(f"Average SSIM: {avg_ssim:.4f}")
    st.write(f"Average PSNR: {avg_psnr:.4f}")
    st.write(f"Slice {slice_idx} SSIM: {slice_ssim:.4f}")
    st.write(f"Slice {slice_idx} PSNR: {slice_psnr:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(real_contrast[0, :, :, slice_idx], caption="Original Enhanced MRI", use_container_width=True)
    with col2:
        st.image(synthetic[0, :, :, slice_idx], caption="Synthetic Enhanced MRI", use_container_width=True)
