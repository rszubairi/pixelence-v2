def decrypt_model_weights(encrypted_file_path: str, decryption_key: str) -> bytes:
    from cryptography.fernet import Fernet
    import os

    if not os.path.exists(encrypted_file_path):
        raise FileNotFoundError(f"The encrypted file {encrypted_file_path} does not exist.")

    with open(encrypted_file_path, 'rb') as encrypted_file:
        encrypted_data = encrypted_file.read()

    fernet = Fernet(decryption_key)
    decrypted_data = fernet.decrypt(encrypted_data)

    return decrypted_data

def save_decrypted_weights(decrypted_data: bytes, output_file_path: str):
    with open(output_file_path, 'wb') as output_file:
        output_file.write(decrypted_data)