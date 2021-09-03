import glob
from PIL import Image
import cv2
from tqdm import tqdm

path = r'your images path here'
c = 0
for file in tqdm(glob.glob('data/plainmulti/vgg_flower/**/**/*.jpg')): 
    img = Image.open(file)
    img = img.resize((84, 84)) #(width, height)
    img.save(file)