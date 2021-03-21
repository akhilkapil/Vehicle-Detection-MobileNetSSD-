import cv2 
import numpy as np 
import matplotlib.pyplot as plt

model = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt','MobileNetSSD_deploy.caffemodel')


blob_height = 300
color_scale = 1.0/127.5
average_color = (127.5, 127.5, 127.5)
confidence_threshold = 0.5

#these are the classes that our model represents(we already have car, bus, person class in it)
labels = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
          'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
          'horse', 'motorbike', 'person', 'potted plant', 'sheep',
          'sofa', 'train', 'TV or monitor']

cap = cv2.VideoCapture('C:\\Users\\akhil\\Downloads\\1615363610851.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX
success, frame = cap.read()
color = (255,170,0)
while success:

    h, w = frame.shape[:2]
    aspect_ratio = w/h

    # Detect objects in the frame.

    blob_width = int(blob_height * aspect_ratio)
    blob_size = (blob_width, blob_height)

    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=color_scale, size=blob_size,
        mean=average_color)

    model.setInput(blob)
    results = model.forward()
    
 
    count = []
    # Iterate over the detected objects.
    for object in results[0, 0]:
        confidence = object[2]
        if confidence > confidence_threshold:
            # Get the object's coordinates.
            
            x0, y0, x1, y1 = (object[3:7] * [w, h, w, h]).astype(int)
            if (x0 <= 1364) & (y0 >=189):

            # Get the classification result.
                id = int(object[1])
                label = labels[id - 1]
    
                # Draw a blue rectangle around the object.
                cv2.rectangle(frame, (x0, y0), (x1, y1),
                              (255, 0, 0), 2)
                
    
                # Draw the classification result and confidence.
                text = '%s (%.1f%%)' % (label, confidence * 100.0)
                cv2.putText(frame, text, (x0, y0 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                count.append(object)
    
    cv2.line(frame, (1,266), (1364,177), color, 2)
    cv2.putText(frame, "vehicles detected: " + str(len(count)), (889, 19), font, 0.6, (0, 180, 80), 2)
    cv2.imshow('Objects', frame)
   

    k = cv2.waitKey(1)
    if k == 27:  # Escape
        break

    success, frame = cap.read()
    
    
    