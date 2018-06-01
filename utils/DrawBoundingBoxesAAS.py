from flask import Flask, Response ,send_file , send_from_directory, request
import cv2
import numpy as np
import os

import json

import math

import png

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


def DrawBoxesOnImage(input_image,boxes,confidence_threshold=-1,write_labels=True):
    (h, w) = input_image.shape[:2]
    
    unique_labels = list(set([str(box["label"]) for box in boxes]))
    print("unique_labels",unique_labels)
    colors = list(np.random.uniform(0, 255, size=(len(unique_labels), 3)))
    label_color_dict = dict(zip(unique_labels, colors + [None] * (len(unique_labels) - len(colors))))

    # loop over the detections
    for box in boxes:
        box_keys = box.keys()
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if ("confidence" not in box_keys) or box["confidence"] > confidence_threshold:
            
            if(write_labels):
                # draw the prediction on the frame
                if(("confidence" in box_keys)):
                    label = box["label"]
                else:
                    label = "{}: {:.2f}%".format(box["label"],box["confidence"] * 100)
            
            startX, startY, endX, endY = GetImageCoordsFromRatios(box,h,w)

            cv2.rectangle(input_image, (startX, startY), (endX, endY),label_color_dict[box["label"]], 2)
            
            y = startY - 15 if startY - 15 > 15 else startY + 15
            
            if(write_labels):
                cv2.putText(input_image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color_dict[box["label"]], 2)

    return input_image


@app.route("/utils/draw_bounding_box", methods=['POST', 'GET'])
def DrawBoxes():
    if request.method == 'POST':
        if 'image' in request.files:
            input_image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)

            json_boxes = json.loads(request.files['boxes'].read())

            drawn_image = DrawBoxesOnImage(input_image,json_boxes["boxes"])

            # cv2.imshow("test",drawn_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            return SendImageArray(drawn_image,convert_color=True)

        if 'image' in request.form.keys():
            input_image = readb64(request.form["image"])
            
            json_boxes = json.loads(request.form['boxes'])

            drawn_image = DrawBoxesOnImage(input_image,json_boxes["boxes"])

            json_data = json.dumps({'image': encIMG64(drawn_image)})
            
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
    app.run(host='0.0.0.0', port=5300)