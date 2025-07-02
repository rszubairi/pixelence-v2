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

def decrypt_file_with_key(encrypted_file_path: str, output_file_path: str, key_path: str):
    """
    Decrypts an encrypted file using a key stored at the specified key path and saves the decrypted content.
    
    Args:
        encrypted_file_path (str): Path to the encrypted file.
        output_file_path (str): Path where the decrypted file will be saved.
        key_path (str): Path to the file containing the decryption key.
    """
    import os
    
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"The key file {key_path} does not exist.")
        
    with open(key_path, 'rb') as key_file:
        decryption_key = key_file.read().decode('utf-8')
        
    decrypted_data = decrypt_model_weights(encrypted_file_path, decryption_key)
    save_decrypted_weights(decrypted_data, output_file_path)
