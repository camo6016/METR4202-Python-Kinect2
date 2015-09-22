import cv2
import numpy as np

'''
haar face detection uses cascadeclassifier to detect objects.
Hand detection is bad. Facedetection is good.
'''

def multiscaledetection(range_min, range_max, scalefactor, detector, minneighbors, deptharray):
        if(scalefactor < 2):
            scalefactor = 2

        temparray = np.float32(deptharray)
        outputarray = []

        for i in range(range_min,range_max,100):
            min_length = i
            max_length = i+150
            scalingfactor = float((max_length-min_length)/256.0)
            temparray = np.float32(deptharray)

            temparray = np.clip(temparray,min_length,max_length)
            temparray = temparray - min_length
            temparray = temparray / scalingfactor
            depthimage = np.uint8(temparray)


            detectoroutput = detector.detectMultiScale(depthimage, scalefactor, minneighbors, minSize = (40,40), maxSize = (200,200))

            

            for (x,y,w,h) in detectoroutput:
                outputarray.append((x,y,w,h))

        return outputarray