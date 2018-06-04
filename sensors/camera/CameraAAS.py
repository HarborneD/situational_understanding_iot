from WebCamera import WebCam
from flask import Flask, Response ,send_file , send_from_directory
import os

import cv2

import base64
from PIL import Image
from StringIO import StringIO

import json


app = Flask(__name__)


def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def encIMG64(image,convert_colour = False):
    if(convert_colour):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    retval, img_buf = cv2.imencode('.jpg', image)
    
    return base64.b64encode(img_buf)


@app.route("/camera_feed/<string:camera_id>", methods=['GET'])
def CameraFeed(camera_id):
	camera = WebCam(int(camera_id))

	response = CameraResponse(camera)
	del(camera)

	return Response(response,mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/camera_update_still/<string:camera_id>", methods=['GET'])
def CameraUpdateStill(camera_id):
	camera = WebCam(int(camera_id))
	camera.SaveStill()
	del(camera)

	return "Done"


@app.route("/camera_still/<string:camera_id>", methods=['GET'])
def CameraStillFile(camera_id):
	path = os.path.join("camera_stills","still_"+camera_id+".jpg")
	return send_file(path, mimetype='image/jpg')


@app.route("/camera_still/as_64string/<string:camera_id>", methods=['GET'])
def CameraStillAsString(camera_id):
	path = os.path.join("camera_stills","still_"+camera_id+".jpg")
	image = cv2.imread(path)

	encoded_image = encIMG64(image)
	response_dict = {"image":encoded_image}
	return json.dumps(response_dict)


@app.route('/camera_stills/<path:path>')
def CameraStill(path):
    return send_from_directory('camera_stills', path)

if __name__ == "__main__":
    print('Starting the API')
    app.run(host='0.0.0.0',port=5100)
