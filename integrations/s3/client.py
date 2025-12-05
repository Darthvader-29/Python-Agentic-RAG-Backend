import os
import uuid
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

def generate_s3_key(filename: str) -> str:
    return f"uploads/{uuid.uuid4()}_{filename}"

def upload_fileobj_to_s3(file_obj, filename: str) -> str:
    key = generate_s3_key(filename)
    s3.upload_fileobj(file_obj, S3_BUCKET_NAME, key)
    return key

def download_s3_to_temp(key: str) -> str:
    tmp_dir = "tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = os.path.join(tmp_dir, key.replace("/", "_"))
    with open(local_path, "wb") as f:
        s3.download_fileobj(S3_BUCKET_NAME, key, f)
    return local_path

def delete_s3_objects(keys: list[str]):
    if not keys:
        return
    s3.delete_objects(
        Bucket=S3_BUCKET_NAME,
        Delete={"Objects": [{"Key": k} for k in keys]},
    )
