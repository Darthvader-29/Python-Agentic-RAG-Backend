import tempfile

import requests

from config import settings

BASE_URL = "https://api.uploadthing.com/v6"


class UploadThingClient:
    def __init__(self):
        api_key = settings.UPLOADTHING_API_KEY
        if not api_key:
            raise ValueError("Missing UPLOADTHING_API_KEY in environment")
        self.headers = {
            "x-uploadthing-api-key": api_key,
            "Content-Type": "application/json",
        }

    def request_presigned_urls(self, files: list):
        """
        Step 1: Request permission to upload files.

        Args:
            files: List of dicts, e.g.:
                   [{"name": "my-file.pdf", "size": 1024, "type": "application/pdf"}]

        Returns:
            List of upload data objects containing 'url', 'key', 'fileUrl', etc.
        """
        endpoint = f"{BASE_URL}/uploadFiles"

        payload = {
            "files": files,
            "acl": "public-read",
            "contentDisposition": "inline",
        }

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()

            data = response.json()

            if not isinstance(data, list):
                raise ValueError(f"Unexpected response format from UploadThing: {data}")

            return data

        except requests.exceptions.HTTPError as e:
            print(f"UploadThing API Error: {e.response.text}")
            raise e
        except Exception as e:
            print(f"Connection Error: {e}")
            raise e

    def delete_files(self, file_keys: list):
        """
        Deletes files from UploadThing to free up space.
        Args:
            file_keys: List of strings (the 'key' we got during upload)
        """
        endpoint = f"{BASE_URL}/deleteFiles"

        payload = {"fileKeys": file_keys}

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to delete files: {e}")
            return False


def download_file_to_temp(file_key: str) -> str:
    """
    Downloads a file from UploadThing's public URL to a local temporary file.

    Args:
        file_key: The unique ID of the file (e.g., "abc-123.pdf")

    Returns:
        str: The absolute path to the temporary file on disk.
    """
    download_url = f"https://utfs.io/f/{file_key}"

    print(f"Downloading from: {download_url}")

    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")

        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)

        temp_file.close()

        return temp_file.name

    except Exception as e:
        print(f"Download failed: {e}")
        raise e


if __name__ == "__main__":
    client = UploadThingClient()
    try:
        test_files = [{"name": "test_document.pdf", "size": 5000, "type": "application/pdf"}]
        result = client.request_presigned_urls(test_files)
        print("SUCCESS! Received Presigned Data:")
        print(result[0])
    except Exception:
        print("FAILED.")
