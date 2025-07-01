import boto3
import os
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def download_file_from_s3(bucket_name, s3_file_key, local_file_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, s3_file_key, local_file_path)
        print(f"Downloaded {s3_file_key} from bucket {bucket_name} to {local_file_path}.")
    except NoCredentialsError:
        print("Credentials not available.")
    except PartialCredentialsError:
        print("Incomplete credentials provided.")
    except Exception as e:
        print(f"Error downloading file: {e}")

def list_files_in_s3_bucket(bucket_name):
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            return [item['Key'] for item in response['Contents']]
        else:
            return []
    except Exception as e:
        print(f"Error listing files in bucket: {e}")
        return []

def check_if_file_exists_in_s3(bucket_name, s3_file_key):
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket_name, Key=s3_file_key)
        return True
    except Exception as e:
        return False

def upload_file_to_s3(local_file_path, bucket_name, s3_file_key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file_path, bucket_name, s3_file_key)
        print(f"Uploaded {local_file_path} to bucket {bucket_name} as {s3_file_key}.")
    except Exception as e:
        print(f"Error uploading file: {e}")