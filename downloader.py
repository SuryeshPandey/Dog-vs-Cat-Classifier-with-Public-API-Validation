import os
import requests
from config import DATASET_URL, IMAGE_DIR

# Custom header to mimic a real browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def download_in_chunks(url, dest_filename, chunk_size=1024*1024):
    # Send a GET request to the URL
    response = requests.get(url, headers=HEADERS, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open the destination file to write the content
        with open(dest_filename, 'wb') as f:
            # Download the file in chunks
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        print(f"✅ Downloaded: {dest_filename}")
    else:
        print(f"❌ Failed to download file. HTTP Status Code: {response.status_code}")

def download_and_extract():
    # Create the image directory if it doesn't exist
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"📁 Created directory: {IMAGE_DIR}")

    # Define the destination file path for the tar.gz file
    tar_filename = os.path.join(IMAGE_DIR, "images.tar.gz")
    print(f"⬇️ Downloading {DATASET_URL} → {tar_filename}")

    # Download the dataset in chunks
    try:
        download_in_chunks(DATASET_URL, tar_filename)

        # Extract the tar.gz file
        import tarfile
        with tarfile.open(tar_filename, 'r:gz') as tar_ref:
            tar_ref.extractall(IMAGE_DIR)
        print(f"✅ Extracted files to {IMAGE_DIR}")

    except Exception as e:
        print(f"❌ Failed to download or extract {DATASET_URL}: {e}")

if __name__ == "__main__":
    download_and_extract()
