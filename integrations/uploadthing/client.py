import os
import requests
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# CONFIGURATION
# Use the 'Secret Key' (sk_live_...) here, NOT the Token
API_KEY = os.getenv("UPLOADTHING_API_KEY") 

if not API_KEY:
    raise ValueError("Missing UPLOADTHING_API_KEY in .env file")

BASE_URL = "https://api.uploadthing.com/v6" #same for all other functions(Delete files, request file access, rename file, etc) but this time only used for uploading

class UploadThingClient:
    def __init__(self):
        self.headers = {
            "x-uploadthing-api-key": API_KEY,
            "Content-Type": "application/json",
            # 'x-uploadthing-version': '6.4.0' # Optional, good practice to match SDK version
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
            "acl": "public-read", # specific to UploadThing's simplified ACL
            "contentDisposition": "inline" # or 'attachment'
        }

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status() # Raises error for 401, 403, 500, etc.
            
            data = response.json()
            
            # Validation: Ensure we got a list back
            if not isinstance(data, list):
                raise ValueError(f"Unexpected response format from UploadThing: {data}")
                
            return data

        except requests.exceptions.HTTPError as e:
            # Print detailed error from UploadThing if available
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
        
        payload = { "fileKeys": file_keys }

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json() # Usually returns { "success": true }
        except Exception as e:
            print(f"Failed to delete files: {e}")
            # We don't raise e here because cleanup failure shouldn't crash the app
            return False

def download_file_to_temp(file_key: str) -> str:
    """
    Downloads a file from UploadThing's public URL to a local temporary file.
    
    Args:
        file_key: The unique ID of the file (e.g., "abc-123.pdf")
        
    Returns:
        str: The absolute path to the temporary file on disk.
    """
    # UploadThing's standard public URL pattern
    download_url = f"https://utfs.io/f/{file_key}"
    
    print(f"Downloading from: {download_url}")
    
    try:
        # Stream=True is good practice for large files
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Create a named temp file that persists until we manually delete it
        # suffix=".pdf" helps PyMuPDF know it's a PDF if it checks extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
        
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
            
        temp_file.close() # Close the handle so other processes can read it
        
        return temp_file.name # Return the path, e.g., "/tmp/tmpxyz123.tmp"
        
    except Exception as e:
        print(f"Download failed: {e}")
        raise e

# --- Usage Example (for testing) ---
if __name__ == "__main__":
    client = UploadThingClient()
    try:
        # Test with a fake file
        test_files = [{
            "name": "test_document.pdf",
            "size": 5000, 
            "type": "application/pdf"
        }]
        result = client.request_presigned_urls(test_files)
        print("SUCCESS! Received Presigned Data:")
        print(result[0])
        # You should see keys like: 'url', 'fields', 'key', 'fileUrl'
    except Exception as e:
        print("FAILED.")
