import argparse
from PIL import Image
import os


def convert_webp_to_png(directory):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(".webp"):
            webp_path = os.path.join(directory, filename)
            png_path = os.path.splitext(webp_path)[0] + ".png"

            # Open the webp file and convert to png
            with Image.open(webp_path) as img:
                img.save(png_path, "PNG")
                print(f"Converted {filename} to {os.path.basename(png_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all WEBP files in a directory to PNG format."
    )
    parser.add_argument(
        "directory", type=str, help="The directory containing WEBP files."
    )
    args = parser.parse_args()

    convert_webp_to_png(args.directory)
