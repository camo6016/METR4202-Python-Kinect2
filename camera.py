from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2
import time

class webcam(object):
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.ID = "kinect"

    def getimage_color(self): #Get cv2 image
        _, frame = self.camera.read()
        return frame

    def getarray_mergeddepth(self):
        return np.zeros((1920,1020,4)) 

    def getimage_mergeddepth(self):
        return np.zeros((1920,1020,4)) 

    def getarray_depth(self):
        return np.zeros((424,512), dtype=np.uint8)

    def getarray_depth_instant(self):
        return np.zeros((424,512), dtype=np.uint8) 

    def getimage_depth(self):
        return np.zeros((424,512), dtype=np.uint8)

    def setrange_depth(self, min_length, max_length):
        pass

    

    
class kinectcamera(object):
    def __init__(self): # This function turns the kinect camera.
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Infrared)

        self.min_length = 500
        self.max_length = 3000
        self.scalingfactor = float(((self.max_length-self.min_length)/256.0))

        self.width_depth = self.kinect.depth_frame_desc.Width
        self.height_depth = self.kinect.depth_frame_desc.Height
        
        self.width_color = self.kinect.color_frame_desc.Width
        self.height_color = self.kinect.color_frame_desc.Height

        self.ID = "kinect"

    def getimage_color(self):
        #Pole untill new frame is avialable
        while True:       
            if self.kinect.has_new_color_frame():
                frame_color = self.kinect.get_last_color_frame()
                break

        #Reshape frame array from 1D list to 3D BGRA array
        outputarray = np.reshape(frame_color,(self.height_color,self.width_color,4))

        return outputarray


    def getimage_mergeddepth(self):
        mergemap = self.kinect.map_depth_to_color()
        mergemap[mergemap>1920] = 0
        mergemap =  np.around(mergemap)

        self.frame_color = self.kinect.get_last_color_frame()
        colorimage = np.reshape(self.frame_color, (self.height_color,self.width_color,4))

        mergedimage = cv2.remap(colorimage, mergemap, None, cv2.INTER_LINEAR)
        
        return mergedimage


    # def getimage_mergeddepth(self):
    #     temparray = self.getarray_mergeddepth()
    #     temparray = np.float32(temparray)

    #     #Clip Frame so that its 8 bit.
    #     temparray = np.clip(temparray,self.min_length,self.max_length)
    #     temparray = temparray - self.min_length
    #     temparray = temparray / self.scalingfactor
    #     arrayframe_depth = np.uint8(temparray)
        
    #     return arrayframe_depth

    def getarray_depth(self):
        #Pole untill new frame is avialable
        while True:       
            if self.kinect.has_new_depth_frame():
                frame_depth = self.kinect.get_last_depth_frame()
                break
            
        #Convert 1D frame array to 2D arrayframe  
        temparray = np.reshape(frame_depth, (self.height_depth,self.width_depth))
        
        return temparray

    def getarray_depth_instant(self):
        self.frame_depth = self.kinect.get_last_depth_frame()
        #Convert 1D frame array to 2D arrayframe  
        temparray = np.reshape(self.frame_depth, (self.height_depth,self.width_depth))
        return temparray
        

    def getimage_depth(self):
        #Pole untill new frame is avialable
        while True:       
            if self.kinect.has_new_depth_frame():
                frame_depth = self.kinect.get_last_depth_frame()
                break


        #Convert 1D frame array to 2D arrayframe  
        temparray = np.reshape(frame_depth, (self.height_depth,self.width_depth))
        temparray = np.float32(temparray)

        #Clip Frame so that its 8 bit.
        temparray = np.clip(temparray,self.min_length,self.max_length)
        temparray = temparray - self.min_length
        temparray = temparray / self.scalingfactor
        arrayframe_depth = np.uint8(temparray)
        #arrayframe_depth = (255-arrayframe_depth)
        
        return arrayframe_depth


    def setrange_depth(self, min_length, max_length): # This function sets the min and max range for the depth camera.
        self.min_length = min_length
        self.max_length = max_length
        self.scalingfactor = float((self.max_length-self.min_length)/256.0)

                
    def end_connection(self): # This ends the connecton with the connect. NOTE:It has never been used before
        self.kinect.close()

