import gdown
import tarfile
import os

url = "https://drive.google.com/uc?id=1W4mSI33IbeKkWztK3XmE05F7m4tNYDYu"
output = "torob-turbo-stage2.tar.gz"

print("Downloading file...")
gdown.download(url, output, quiet=False)

print("Extracting file...")
with tarfile.open(output, "r:gz") as tar:
    tar.extractall("torob-turbo-stage2")
    print(f"Extracted to {os.path.abspath('torob-turbo-stage2')}")