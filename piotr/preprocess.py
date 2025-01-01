from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import imageio
import pathlib
import cv2
import numpy as np
from tqdm import tqdm

target_res = 512

root = "/Users/piotrblaszyk/Documents/university/year4/ibm-z-datathon/archive-3/the_wildfire_dataset/the_wildfire_dataset"
new_root = f"/Users/piotrblaszyk/Documents/university/year4/ibm-z-datathon/archive-3/the_wildfire_dataset/the_wildfire_dataset-{target_res}"

root_path = pathlib.Path(root)
new_root_path = pathlib.Path(new_root)
imgs = list(root_path.glob("**/*.jpg"))
# short = random.sample(imgs, 4)

for i, img_path in tqdm(enumerate(imgs)):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, dsize=(target_res, target_res))
    im = Image.fromarray(img)

    rel_path = img_path.relative_to(root_path)
    parent = f"{new_root}/{str(rel_path)}"
    parent = pathlib.Path(parent).parent
    parent.mkdir(parents=True, exist_ok=True)

    new_file_path = f"{new_root}/{str(rel_path)}"
    im.save(new_file_path)
