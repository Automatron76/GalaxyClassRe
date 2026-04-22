"""
Download SDSS galaxy cutout images.

For each galaxy in the labels CSV file, this script requests a 256×256 px
JPEG cutout from the SDSS SkyServer DR14 Image Cutout API using the
galaxy's RA/Dec coordinates.

So in the .gz file we have technical data and in the SDSS SkyServer we have
the corresponding images of that data.

Image source: SDSS SkyServer DR14 Image Cutout API
https://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg

"""
import os
import requests
import argparse
import pandas as pd
from time import sleep
from config import LABELS_PATH, IMAGES_DIR

# SDSS SkyServer Image Cutout endpoint (Data Release 14) https://skyserver.sdss.org/dr14/en/help/docs/api.aspx
SDSS_CUTOUT_URL = "https://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg"

# Image parameters from the SDSS SkyServer API documentatio
SCALE = 0.2        # arcseconds per pixel
WIDTH = 256        # pixels
HEIGHT = 256       # pixels

REQUEST_DELAY = 0.3

def download_galaxy_image(objid: str, ra: float, dec: float, output_dir: str) -> bool:
    out_path = os.path.join(output_dir, f"{objid}.jpg")

    if os.path.exists(out_path):
        return True
    

    url = (
        f"{SDSS_CUTOUT_URL}"
        f"?ra={ra}&dec={dec}&scale={SCALE}"
        f"&width={WIDTH}&height={HEIGHT}"
    )
        
    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(response.content)
            return True
        return False

    except Exception as e:
        print(f"Error: {e}")
        return False
    
    def main():
        # Set up --limit argument, default 5000
        parser = argparse.ArgumentParser()
        parser.add_argument("--limit", type=int, default=5000)
        args = parser.parse_args()

        # Create images folder if it doesn't exist
        os.makedirs(IMAGES_DIR, exist_ok=True)

        # Read only the first N rows from the labels CSV
        df = pd.read_csv(LABELS_PATH, dtype={"id": "string"}).head(args.limit)