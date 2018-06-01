from WebCamera import WebCam
from flask import Flask, Response ,send_file , send_from_directory
import os


app = Flask(__name__)


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

@app.route('/camera_stills/<path:path>')
def CameraStill(path):
    return send_from_directory('camera_stills', path)

if __name__ == "__main__":
    print('Starting the API')
    app.run(host='0.0.0.0',port=5100)
