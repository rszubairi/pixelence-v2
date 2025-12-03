# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:03:56 2024

@author: amraa
"""

import streamlit as st
import pydicom
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from mod_SRGAN_DeepRes import build_generator


# Set Streamlit to fullscreen mode
st.set_page_config(layout="wide")

# Set the title of the Streamlit app
st.title("Image Browser and Viewer")

# Add custom CSS to set the sidebar width, center the logo, and add padding to sliders
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 300px;
    }
    .sidebar-content {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .sidebar-padding {
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

lw_x = 128
h_x = lw_x * 4
generated_image = None
### build necessary Function
def generate_hr_image(image):
    generator = build_generator((lw_x, lw_x,1))
    generator.load_weights('weight - generator_res128 - Good.h5')
    
    #if generator  is not None:
        #st.write('## ** SuperRes Engine Is ready **')
    #    st.markdown('<h2 style="color:red;">SuperRes Engine Is Initiated Successfully</h2>', unsafe_allow_html=True)

    im_hr = image.copy()
    im_hr = normalisation_float32(im_hr)
    im_hr = generator.predict(np.expand_dims(im_hr, 0))
    im_hr = np.squeeze(im_hr, 0)
    im_hr = (im_hr + 1) / 2
    im_hr = im_hr * 255
    return im_hr.astype('uint8')
    
def convert_uint8(image):
   if image.dtype != np.uint8:
       image = (image / np.max(image) * 255).astype(np.uint8)
   return image

def normalisation_float32(image):
        image = image.astype('float32')
        min_val = np.min(image)
        max_val = np.max(image)
  
        # Normalize the pixel values
        epsilon = 1e-7
        image = (image - min_val) / (max_val - min_val + epsilon)
        image = image * 2 - 1
        
        return image 
    
def convert_float32(image):
    if image.dtype != np.float32:
        image = image.astype(np.float32)
        image /= np.max(image)  # Normalize to range [0, 1]
    return image

def lower_res_add_noise(image, noise_magnitude = 0.0, blue_magnitude = 0.0):
    low_res = convert_float32(image.copy())

    print(low_res.shape)
    def add_gaussian_noise(image, mean=0, std=noise_magnitude):
        # Generate Gaussian noise
        noise = np.random.normal(mean, std, image.shape)
        # Add the noise to the image
        noisy_image = image + noise
        # Clip values to be in the valid range [0, 1]
        noisy_image = np.clip(noisy_image, 0, 1)
        return noisy_image

    def apply_gaussian_blur(image):      
       # Apply Gaussian blur using scipy's gaussian_filter
       blurred_image = gaussian_filter(image, sigma=blue_magnitude)
    
       return blurred_image

    low_res = apply_gaussian_blur(low_res)
    low_res = add_gaussian_noise(low_res)

    
    if len(low_res.shape) < 3:
        low_res = np.expand_dims(low_res, -1)
    return zoom(convert_uint8(low_res), (lw_x/low_res.shape[0], lw_x/low_res.shape[1], 1))

# Add a sidebar with a logo
logo_path = "Logo-Black.png"  # Replace with your logo image file path
st.sidebar.image(logo_path, use_column_width=False, width=150)
st.sidebar.write("## **BrainSR V1.0.3 FROM Pixellence**")
st.sidebar.write("This is the DICOM Image Viewer. You can adjust the sliders and upload a DICOM file to view the noisy and super-resolution images side by side.")

    
### upload dcm
uploaded_file = st.file_uploader("Please Upload the low res scan...", type=["dcm"])


# Check if a file is uploaded
if uploaded_file is not None:
    scan = pydicom.dcmread(uploaded_file)
    scan_im = scan.pixel_array
    
    # Create sliders in the sidebar
    slider1_val = st.sidebar.slider('Add Random Noise', min_value=0, max_value=5, value=0) / 100
    slider2_val = st.sidebar.slider('Add Blur', min_value=0, max_value=5, value=0) / 10

    if st.sidebar.button('Show SuperRes Image'):
        show_superres = True
    else:
        show_superres = False

    # Display two images side by side
    col1, col2 = st.columns(2)
    

    # Display the DICOM image twice (side by side)
    if scan_im is not None:
        with col1:        
            low_res_image = lower_res_add_noise(scan_im, slider1_val, slider2_val)
            
            ## ensure correct dim plotting
            hr_shape = (lw_x,lw_x,1)
            lr_shape = (h_x, h_x,1)
            
            corrected_low_res_img = np.ones(hr_shape, dtype=np.uint8)
            start_x = (hr_shape[0] - lr_shape[0]) // 2
            start_y = (hr_shape[1] - lr_shape[1]) // 2
            
            corrected_low_res_img[start_x:start_x+lr_shape[0], start_y:start_y+lr_shape[1]] = low_res_image

            st.image(low_res_image, caption='Noisy Image.', use_column_width=True)
            
        if show_superres:
            st.sidebar.write('<h3 style="color:red;">SuperRes Engine Is Initiated Successfully</h3>', unsafe_allow_html=True)

            with col2:
                generated_image = generate_hr_image(low_res_image)
                st.image(generated_image, caption='SuperRes Image.', use_column_width=True)
else:
    st.write("Please upload a DICOM file.")
