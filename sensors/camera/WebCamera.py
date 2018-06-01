import cv2
import os
import time
import datetime
from shutil import copyfile

class WebCam():
  def __init__(self, camera_index=1):
  	self.camera_index = camera_index
  	self.file_name = "still_"+str(self.camera_index)+".jpg"
  	try:
  		print("creating camera")
  		self.source = cv2.VideoCapture(camera_index)
  		print(type(self.source))
  	except:
  		print("error creating camera")
  	
  def __del__(self):
  	self.source.release()
  
  def GetFrame(self):
  	ret, jpeg = cv2.imencode('.jpg',self.GetFrameArray())
  	return jpeg.tobytes()
  
  def GetFrameArray(self):
  	success, image = self.source.read()
  	return image
  
  def SaveStill(self,path=""):
  	if(path == ""):
  		path = os.path.join("camera_stills",self.file_name)
  	try:
  		print("saving still")
  		return_value,image = self.source.read()
  		cv2.imwrite(path,image)
  	except:
  		print("error saving still")
  	



if __name__ == '__main__':
	camera = WebCam(0)
	camera.SaveStill("test2.jpg")
	del(camera)
	# image = camera.GetFrameArray()
	# cv2.imwrite(os.path.join("test.jpg"),image)
	# time_stamp = time.time()
	# st = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H-%M-%S')
	# new_path = os.path.join("camera_stills",st+".jpg")
	# path = os.path.join("still_0.jpg")
	# # while True:
	# 	time.sleep(1)
	# 	time_stamp = time.time()
	# 	camera.SaveStill(path)

    #st = datetime.datetime.fromtimestamp(time_stamp).strftime('%d-%H-%M-%S')
    #new_path = os.path.join("camera_stills",st+".jpg")
    #image = camera.GetFrameArray()
    #cv2.imwrite(new_path,image)
    #copyfile(new_path, path)
    #cv2.imshow("web cam",image)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #	break
