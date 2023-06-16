import argparse
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import random
from src.gen_polygon import multi_image_augment
from src.img_convert import insert_bg, translate_image
from PIL import Image
from io import BytesIO
import base64

def PIL_to_encode(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string

def main():
        
    parser = argparse.ArgumentParser(description='Augment Multi Image from Labelme json + img and Background img')
    parser.add_argument('-n','--samplenum', type=int, help='Number of output samples')
    parser.add_argument('-i', '--jsondir', type=str, help='Input folder with labelme jsonfile and image ')
    parser.add_argument('-b','--bgdir', type=str, help='Input folder with background image')
    parser.add_argument('-o','--outdir', type=str, help='Output folder with images and json files')
    parser.add_argument('--min', type=int, default=2, help='Min objects in an image')
    parser.add_argument('--max', type=int, default=4, help='Max objects in an image')
    parser.add_argument('--dataname', type=str, default="merge", help='name of out put. ex: "buscard" -> "buscard_0001.jpg"' )

    args = parser.parse_args()

    jsondir = args.jsondir

    json_list = [x for x in os.listdir(jsondir) if "json" in x]

    bg_list = []
    print("Background loading...")
    for filename in tqdm(os.listdir(args.bgdir)):
        image = cv2.cvtColor(cv2.imread(os.path.join(args.bgdir, filename)), cv2.COLOR_BGR2RGB)
        bg_list.append(image)
    
    print("Augment process...")
    for i in tqdm(range(1, 1+args.samplenum)):
        list_polygon, w, h, json_paths = multi_image_augment(jsondir, json_list, num_img = random.randint(args.min, args.max))

        new_image = cv2.resize(random.choice(bg_list), (w, h))
        new_polygons = []
        for j in  range(len(list_polygon)):
            with open(json_paths[j], "r") as f:
                json_data = json.load(f)
            image_path = os.path.join(jsondir, json_data['imagePath'])
            polygon1 = np.array(json_data['shapes'][0]['points']).astype(np.int32)
            polygon2 = list_polygon[j].astype(np.int32)

            translated_image, last_polygon = translate_image(image_path, polygon1, polygon2, w, h)
            new_polygons.append(last_polygon)
            new_image = insert_bg(translated_image, new_image, last_polygon, h, w)
        # json_data keys: (['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth'])
        new_name = f"{args.dataname}_{i:05d}"
        shapes=[]
        for polygon in new_polygons:
            shape = {
                'label':'0', 
                'points': polygon.tolist(), 
                'group_id': json_data['shapes'][0]['group_id'], 
                'shape_type': 'polygon', 
                'flags': json_data['shapes'][0]['flags']
            }
            shapes.append(shape)
        pil_img = Image.fromarray(new_image)
        json_data['shapes'] = shapes
        json_data['imagePath'] = f"{new_name}.jpg"
        json_data['imageHeight'] = h
        json_data['imageWidth'] = w
        json_data['imageData'] = PIL_to_encode(pil_img)

        pil_img.save(os.path.join(args.outdir, json_data['imagePath']))
        with open(os.path.join(args.outdir, f"{new_name}.json"), "w") as out:
            json.dump(json_data, out)

if __name__ == "__main__":
    main()
