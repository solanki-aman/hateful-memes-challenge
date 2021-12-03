import pandas as pd
from PIL import Image
import PIL

image_path = '/Users/amansolanki/datasets/hateful-memes-images/'
image_save_path = '/Users/amansolanki/PycharmProjects/hateful-memes-challenge/demo/demo_data/images/'
test_seen_original = pd.read_csv('/Users/amansolanki/PycharmProjects/hateful-memes-challenge/data/test_seen.csv')

demo_data = test_seen_original.sample(2, random_state=7)

# Save Images
for image in demo_data['image_id']:
    picture = Image.open(image_path+image)
    picture = picture.save(image_save_path+image)

# Save CSV
demo_data.to_csv('demo_data.csv', index=False)
