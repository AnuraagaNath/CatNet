import os
import imageio.v2 as imageio
from glob import glob
import re

# Folder containing the saved images
input_folder = './result_images'
output_video = 'output_video.mp4'

# Get the list of image files
image_files = sorted(glob(input_folder + '/*'), key=lambda x: int(re.search(r"(\d+)", x).group(1)))

# Create video
with imageio.get_writer(output_video, fps=30) as writer:
    for image_file in image_files:
        image = imageio.imread(image_file)
        writer.append_data(image)
