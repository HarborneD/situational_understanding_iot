from flask import Flask, Response ,send_file , send_from_directory, request
import cv2
import numpy as np
import os

import json

import math

import png

import base64
from PIL import Image
from StringIO import StringIO


app = Flask(__name__)


def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def encIMG64(image,convert_color = False):
    if(convert_color):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    retval, img_buf = cv2.imencode('.jpg', image)
    
    return base64.b64encode(img_buf)

def SendImageArray(image_array,convert_color = False):
    if(convert_color):
        image_array = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
    
    png.from_array(image_array, 'RGB').save("tmp.png")
    
    return send_file("tmp.png")


def GetImageCoordsFromRatios(box,image_height,image_width):
    x1 = int(math.floor(box["startX"] * image_width))
    x2 = int(math.floor(box["endX"] * image_width))
    y1 = int(math.floor(box["startY"] * image_height))
    y2 = int(math.floor(box["endY"] * image_height))

    return x1, y1, x2, y2


def GetSubImages(input_image,boxes,fetch_labels=[]):
    (h, w) = input_image.shape[:2]
    
    # loop over the detections
    sub_images = {}
    for box in boxes:
        if(fetch_labels == [] or box["label"] in fetch_labels):

                
            startX, startY, endX, endY = GetImageCoordsFromRatios(box,h,w)

            if(len(input_image.shape) == 3):
                sub_image = input_image[startY:endY,startX:endX,:]
            else:
                sub_image = input_image[startY:endY,startX:endX]
            
            if(not box["label"] in sub_images.keys()):
                sub_images[box["label"]] = []
            sub_images[box["label"]].append(sub_image)
        
    return sub_images

def EncodeSubImageDict(sub_image_dict):
    keys = list(sub_image_dict.keys())
    keys.sort()

    encoded_sub_image_dict = {}
    for label in keys:
        encoded_sub_image_dict[label] = []

        for image in sub_image_dict[label]:
            encoded_sub_image_dict[label].append(encIMG64(image,True))

    return encoded_sub_image_dict


@app.route("/utils/extract_sub_images", methods=['POST', 'GET'])
def ExtractImages():
    if request.method == 'POST':
        if 'image' in request.files:
            input_image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)

            json_boxes = json.loads(request.files['boxes'].read())

            drawn_image = DrawBoxesOnImage(input_image,json_boxes["boxes"])

            if 'sub_image_labels' in request.form.keys():
                target_labels = json.loads(request.form['sub_image_labels'])["sub_image_labels"]
            else:
                target_labels = []

            sub_image_dict = GetSubImages(input_image,json_boxes["boxes"],target_labels)

            # cv2.imshow("test",drawn_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            print("sending first sub image")
            return SendImageArray(sub_image_dict[list(sub_image_dict.keys())[0]][0],convert_color=True)

        if 'image' in request.form.keys():
            input_image = readb64(request.form["image"])
            
            json_boxes = json.loads(request.form['boxes'])

            if 'sub_image_labels' in request.form.keys():
                target_labels = json.loads(request.form['sub_image_labels'])["sub_image_labels"]
            else:
                target_labels = []

            sub_image_dict = GetSubImages(input_image,json_boxes["boxes"],target_labels)

            json_data = json.dumps({'sub_images': sub_image_dict})
            # json_data = json.dumps({'sub_images': EncodeSubImageDict(sub_image_dict)})
            
            return json_data


        return 'error'
    return '''
    <!doctype html>
    <title>Upload Image and Boxes Files to Draw Bounding Boxes on Image</title>
    <h1>Upload Image and Boxes Files to Draw Bounding Boxes on Image</h1>
    <form method=post enctype=multipart/form-data>
    <p><label for="image">Image:</label><input type=file name=image>
    <p><label for="boxes">Box File:</label><input type=file name=boxes>
    <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    print('Starting the API')
    app.run(host='0.0.0.0', port=5310)