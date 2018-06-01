### based on code from: https://www.pyimagesearch.com/2017/10/16/raspberry-pi-deep-learning-object-detection-with-opencv/    accessed:01/06/2018


import numpy as np
import cv2

import argparse
import imutils
import time

class PiObjectDetector(object):
	"""docstring for PiObjectDetector"""
	def __init__(self, prototxt_path, model_path):
		super(PiObjectDetector, self).__init__()
		self.model_path = model_path
		self.prototxt_path = prototxt_path
		
		self.InitialiseClassesAndColours()

		# load our serialized model from disk
		print("[INFO] loading model...")
		self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)


	def InitialiseClassesAndColours(self):
		# initialize the list of class labels MobileNet SSD was trained to
		# detect, then generate a set of bounding box colors for each class
		self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]
		self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))


	def DetectObjectsFromPath(self,image_path,confidence_threshold=-1,as_dict = False):
		img = cv2.imread(image_path)
		return self.DetectObjectsFromArray(img,confidence_threshold,as_dict)


	def DetectObjectsFromArray(self,image,confidence_threshold=-1,as_dict = False):
		#detections in the form [? class_id confidence x1_ratio y1_ratio x2_ratio y2_ratio] 
		#'ratio' = cordinate in the image as represented by how muchof the total dimension it is locatated at (as a float 0 to 1)
		
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),0.007843, (300, 300), 127.5)
		# pass the blob through the network and obtain the detections and
		# predictions
		self.net.setInput(blob)

		detections = self.net.forward()

		if(confidence_threshold>0):
			detections = self.FilterDetectionsByConfidence(detections,confidence_threshold)

		if(as_dict):
			detections = self.DetectionsToDicts(detections)

		return detections


	def FilterDetectionsByConfidence(self,detections,confidence_threshold=0.8):
		filtered_detections=[]
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
		
			if confidence > confidence_threshold:
				filtered_detections.append(detections[:,:,i,:])
		print("filtered_detections",len(filtered_detections))
		return np.array(filtered_detections)


	def DrawDetections(self, input_image,detections,confidence_threshold=-1):
		(h, w) = input_image.shape[:2]
		
		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > confidence_threshold:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(self.CLASSES[idx],confidence * 100)
				cv2.rectangle(input_image, (startX, startY), (endX, endY),self.COLORS[idx], 2)
				
				y = startY - 15 if startY - 15 > 15 else startY + 15
				
				cv2.putText(input_image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

		return input_image


	def DetectionsToDicts(self,detections):
		detection_dicts_list = []
		detection_list = detections[:,0,0,:]
		
		for detection in detection_list:
			detection_dicts_list.append(self.DetectionToDict(detection))

		return detection_dicts_list


	def DetectionToDict(self,detection):
		idx = int(detection[1])
		class_name = self.CLASSES[idx]
		confidence = float(detection[2])
		box = detection[3:7]
		(startX, startY, endX, endY) = box.astype("float")
		
		return {"label":class_name,"confidence":confidence,"startX":startX,"startY":startY,"endX":endX,"endY":endY}


if __name__ == '__main__':
	image_path = "index.jpg"

	detector = PiObjectDetector(prototxt_path="MobileNetSSD_deploy.prototxt.txt", model_path="MobileNetSSD_deploy.caffemodel")

	frame = cv2.imread(image_path)

	detections = detector.DetectObjectsFromPath(image_path)

	bounded_img = detector.DrawDetections(frame,detections,0.2)

	cv2.imshow('dst_rt', bounded_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	detections = detector.DetectObjectsFromArray(frame)

	print(detections)

	detections = detector.FilterDetectionsByConfidence(detections,0.9)

	print(detections)	

	print(detector.DetectionsToDicts(detections))