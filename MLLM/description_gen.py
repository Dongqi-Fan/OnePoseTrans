import os
import google.generativeai as genai
import time
from PIL import Image
from tqdm import tqdm

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


os.environ["API_KEY"] = "YOU KEY"
genai.configure(api_key=os.environ['API_KEY'])
model = genai.GenerativeModel('gemini-1.5-pro', safety_settings=safety_settings)

with open('template.txt', 'r') as f:
       template = f.readlines()
prompt = f"{' '.join(template)}"

images_path = 'YOU IMAGE PATH'
index = 0
for img_name in tqdm(sorted(os.listdir(images_path))):
    image = Image.open(images_path + img_name)
    response = model.generate_content([prompt, image])
    result = response.text
    with open('wpose_descriptions.txt', 'a') as f:
        f.writelines('{:5d}/{}/{}'.format(index, img_name, result))
    index += 1
    time.sleep(3)



