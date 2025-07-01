import torch
import os
from s3_utils import download_file_from_s3
from decrypt_utils import decrypt_weights

def load_model_weights(model, weight_path, s3_bucket=None, s3_key=None, decrypt_key=None):
    if s3_bucket and s3_key:
        try:
            # Attempt to download and decrypt weights from S3
            encrypted_weight_path = download_file_from_s3(s3_bucket, s3_key)
            decrypted_weight_path = decrypt_weights(encrypted_weight_path, decrypt_key)
            model.load_state_dict(torch.load(decrypted_weight_path))
            print("Loaded weights from S3.")
            return
        except Exception as e:
            print(f"Failed to load weights from S3: {e}")

    # Fallback to local weights
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print("Loaded weights from local path.")
    else:
        raise FileNotFoundError(f"Weight file not found at {weight_path}")

def run_inference(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output

# Example usage
if __name__ == "__main__":
    # Assuming 'generator' is your model instance
    weight_path = "/Users/raheelzubairi/Documents/projects/pixelence/v2/_weights_/_global1.pth"
    s3_bucket = "your-s3-bucket-name"
    s3_key = "path/to/encrypted/weights.pth"
    decrypt_key = "your-decryption-key"

    load_model_weights(generator, weight_path, s3_bucket, s3_key, decrypt_key)

    # Further inference code can be added here.