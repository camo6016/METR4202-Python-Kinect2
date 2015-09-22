import numpy as np
import time
import cv2
import math
import glob
import pickle
import os.path
from prettytable import PrettyTable

from camera import *
from haardetection import *
from projection import *


# Calibration
global mouseclick
mouseclick = []

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseclick.append((x,y))


def main():
    camera = kinectcamera()
    camera.setrange_depth(300,1200)
    #camera = webcam()
    referanceframe = pose()
    projection = translator((370,370),(256,212),424,512)
       
    #Perform Calibration
    try:
        file = open('calibrationdata.txt', 'r')
        calibrationarray = pickle.load(file)
        mtx = calibrationarray[0]
        dist = calibrationarray[1]
        newcameramtx = calibrationarray[2]
        roi = calibrationarray[3]
        file.close()

        print("Camera Calibration Coeeficients")
        print("Focal Lengths   -- Fx: " + str(newcameramtx[0,0]) + "  Fy: " + str(newcameramtx[1][1]))
        print("Principal Point -- Cx: " + str(newcameramtx[0,2]) + "  Cy: " + str(newcameramtx[1][2]))
        print("Distortion      -- Kc: " + str(dist))

    except:
        print("Unable to find calibration data")
        print("Calibrate Camera Immediately")



    #Start the Main Loop
    state = "color"
    filenameposition = 0
    counter=0
    start = time.clock()
    global mouseclick
    while(1):
        if(state == "depth"):
            range_min = cv2.getTrackbarPos('depthcamerarangemax','Depth Image')
            range_max = cv2.getTrackbarPos('depthcamerarangemin','Depth Image')
            camera.setrange_depth(range_min,range_max)
            depthimage = camera.getimage_depth()

        if(state == "posedetection"):
            colorimage = camera.getimage_mergeddepth()
            deptharray = camera.getarray_depth()
            outputvector, rotationvector, flag, corners, imgpts = referanceframe.findpose(colorimage, deptharray, newcameramtx, dist)
            if(flag==True):
                storecorners = corners
                storeimgpts = imgpts
                storeoutputvector = outputvector
                storerotationvector = rotationvector
            if 'storecorners' in locals():
                colorimage = referanceframe.draw(colorimage, storecorners, storeimgpts)
                P_c = projection.calculate(storeoutputvector)
                cv2.putText(colorimage, str(P_c[0]) + " " + str(P_c[1]) + " " + str(P_c[2]) + " " , (storeoutputvector[0],storeoutputvector[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255))
                cv2.circle(colorimage, (255,212), 2, (0,0,255), 5)



        if(state == "color"): 
            colorimage = camera.getimage_color()
            #colorimage = camera.getimage_mergeddepth()
            colorimage = cv2.resize(colorimage,(960, 510))
            #colorimage = cv2.undistort(colorimage, mtx, dist, None, newcameramtx)

        if(state == "calibration"):
            time.sleep(2)
            colorimage = camera.getimage_color()
            colorimage = cv2.resize(colorimage,(960, 510)) 
            gray = cv2.cvtColor(colorimage,cv2.COLOR_BGR2GRAY)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.
            ret = True
            while(ret==True):
                    x=4
                    y=3
                    objp = np.zeros((x*y,3), np.float32)
                    objp[:,:2] = np.mgrid[0:x,0:y].T.reshape(-1,2)
                    ret, corners = cv2.findChessboardCorners(gray, (x,y), None, 11)
                    if(ret == True):
                        #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                        x_min = int(round(np.amin(corners[:,0,0])))
                        y_min = int(round(np.amin(corners[:,0,1])))
                        x_max = int(round(np.amax(corners[:,0,0])))
                        y_max = int(round(np.amax(corners[:,0,1])))
                        cv2.rectangle(gray, (x_min, y_min), (x_max, y_max), (255,255,255), -1)
                        cv2.rectangle(colorimage, (x_min, y_min), (x_max, y_max), (255,255,255), -1)
                        colorimage = cv2.drawChessboardCorners(colorimage, (x,y), corners,ret)
                        objpoints.append(objp)
                        imgpoints.append(corners)   
            if(len(objpoints)>10):            
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
                h,  w = colorimage.shape[:2]
                newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
                print("Camera Calibration Coeeficients")
                print("Focal Lengths   -- Fx: " + str(newcameramtx[0,0]) + "  Fy: " + str(newcameramtx[1][1]))
                print("Principal Point -- Cx: " + str(newcameramtx[0,2]) + "  Cy: " + str(newcameramtx[1][2]))
                print("Distortion      -- Kc: " + str(dist))
                print(" ")

                file = open('calibrationdata.txt', 'w')
                pickle.dump([mtx,dist,newcameramtx,roi], file)
                file.close()

             
        if(state == "imagecolection"):
            filename = str(filenameposition) + ".jpg"
            if(os.path.isfile(filename)):
                filenameposition=filenameposition+1
                continue
            if((time.clock()-start)>1):
                start = time.clock()

                crop2 = camera.getimage_color()
                crop2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
                crop = cv2.resize(crop2,(960, 510))
                depthimage = crop

                # camera.setrange_depth(500,1000)
                # depthimage = camera.getimage_depth()

                # _,thresh = cv2.threshold(depthimage,1,255,cv2.THRESH_BINARY)

                # contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
                # position = 0
                # largestcontour = 0
                # if(len(contours)==0):
                #     continue
                # for i in range(len(contours)):
                #     area = cv2.contourArea(contours[i])
                #     if(area>largestcontour):
                #         largestcontour = area
                #         position = i
                #         #print(area)
                #         #print(position)
                # cnt = contours[position]            
                # x,y,w,h = cv2.boundingRect(cnt)
                # crop = depthimage[y:y+h,x:x+w]

                #depthimage = cv2.cvtColor(depthimage, cv2.COLOR_BGR2GRAY)
                #depthimage = cv2.resize(depthimage,(960, 510))
                cv2.imwrite(filename,crop)
                print(filename)


        if(state == "colorcalibration"): # Manual Color Calibration
            start = time.time()
            while((time.time()-start)<3):
                colorimage = camera.getimage_color()
                colorimage = cv2.resize(colorimage,(960, 510))
                position = round((time.time()-start),1)
                cv2.putText(colorimage, str(position), (300,255), cv2.FONT_HERSHEY_TRIPLEX, 5, (0,0,255))
                cv2.imshow('Color Calibration', colorimage)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:         # wait for ESC key to exit
                    cv2.destroyAllWindows()
                    print("Color")
                    state = "color"
                    mouseclick = []
                    break

            colorimage = camera.getimage_color()
            colorimage = cv2.resize(colorimage,(960, 510))
            cv2.setMouseCallback('Color Calibration',draw_circle)
            while(len(mouseclick)<4):
                cv2.imshow('Color Calibration', colorimage)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:         # wait for ESC key to exit
                    cv2.destroyAllWindows()
                    print("Color")
                    state = "color"
                    mouseclick = []
                    break

            pts1 = np.float32([[mouseclick[0][0],mouseclick[0][1]],[mouseclick[1][0],mouseclick[1][1]],[mouseclick[2][0],mouseclick[2][1]],[mouseclick[3][0],mouseclick[3][1]]])
            pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(colorimage,M,(300,300))
            dsthsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
            dstlab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)

            colorname = ["Dark Skin ","Light Skin ","Blue Sky ","Foliage ","Blue Flower ","Bluish Green ", \
            "Orange ","Purplish ","Moderate Red ","Purple ", "Yellow Green ", "Orange Yellow ", \
            "Blue ", "Green ","Red ", "Yellow ", "Magenta ", "Cyan ", \
            "White ", "Grey 1 ", "Grey 2 ", "Grey 3", "Grey 4 ", "Black ",]

            colorposition = [[22,35], [75,35], [124,35], [175,35], [225,35], [275,35], \
            [22,115], [75,115], [124,115], [175,115], [225,115], [275,115], \
            [22,190], [75,190], [124,190], [175,190], [225,190], [275,190], \
            [22,270], [75,270], [124,270], [175,270], [225,270], [275,270]]

            propervalue = [[38,12,14], [66,13,17], [51,0,-22], [43,-17,22], [56,13,-25], [72,-31,1], \
            [62,28,58], [41,18,-43], [52,43,15], [31,26,-24], [72,-28,59], [72,12,67], \
            [30,27,-51], [55,-41,34], [41,51,26], [81,-4,79], [52,49,-16], [52,-22,-27], \
            [96,0,0], [81,0,0], [67,0,0], [52,0,0], [36,0,0], [20,0,0]]

            outputtable = PrettyTable(['Color Number', 'Color Name', 'RGB Color Space', "HSV Color Space", "LAB Color Space", "Expected LAB Value"])

            for i in range(len(colorname)):
                x = colorposition[i][0]-15
                y = colorposition[i][1]-15
                mean = np.uint8(np.around(cv2.mean(dst[x:x+30, y:y+30]))[:3])
                meanhsv = np.uint8(np.around(cv2.mean(dsthsv[x:x+30, y:y+30]))[:3])
                meanlab = np.uint8(np.around(cv2.mean(dstlab[x:x+30, y:y+30]))[:3])
                outputtable.add_row([i+1, colorname[i], mean, meanhsv, meanlab, [propervalue[i][0], (propervalue[i][1]+128), (propervalue[i][2]+128)]])
                cv2.circle(dst, (x,y), 1, (38,14,16), 2)
                cv2.circle(dst, (x+30,y+30), 1, (38,14,16), 2)

            print(outputtable)

            while(1):
                cv2.imshow('Color Calibration', dst)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:         # wait for ESC key to exit
                    cv2.destroyAllWindows()
                    print("Color")
                    state = "color"
                    mouseclick = []
                    break

        if(state == "positiondetection"):
            colorimage = camera.getimage_mergeddepth()
            deptharray = camera.getarray_depth()
            outputimage = camera.getimage_depth()

            outputvector, rotationvector, flag, corners, imgpts = referanceframe.findpose(colorimage, deptharray, newcameramtx, dist)
            if(flag==True):
                storecorners = corners
                storeimgpts = imgpts
                storeoutputvector = outputvector
                storerotationvector = rotationvector

            if 'storecorners' in locals():
                colorimage = referanceframe.draw(colorimage, storecorners, storeimgpts)
                P_c = projection.calculate(storeoutputvector)
                cv2.putText(colorimage, str(P_c[0]) + " " + str(P_c[1]) + " " + str(P_c[2]) + " " , (storeoutputvector[0],storeoutputvector[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255))
                cv2.circle(colorimage, (255,212), 2, (0,0,255), 5)


                range_min = cv2.getTrackbarPos('depthcamerarangemax','Position Detection')
                range_max = cv2.getTrackbarPos('depthcamerarangemin','Position Detection')
                scalefactor = cv2.getTrackbarPos('scalingfactor','Position Detection')
                minneighbors = cv2.getTrackbarPos('minimumneighbors','Position Detection')
                if(scalefactor < 2):
                    scalefactor = 2


                detectoroutput = multiscaledetection(range_min, range_max, scalefactor, detector, minneighbors, deptharray)

                for (x,y,w,h) in detectoroutput:
                    cv2.rectangle(colorimage,(x,y),(x+w,y+h),(255,0,0),2)
                    detecteddistance = deptharray[x+w/2, y+h/2]


                    detectedpositonvector = projection.calculate([x+w/2, y+h/2, detecteddistance])

                    P_h = np.mat([[detectedpositonvector[0]],[detectedpositonvector[1]],[detectedpositonvector[2]],[0]])

                    #Show Location on image
                    if(detecteddistance != 0):
                        originstring = str(detectedpositonvector[0]) + ", " + str(detectedpositonvector[1]) + ", " + str(detectedpositonvector[2])
                    else:
                        originstring = "Too close to camera"
                    
                    cv2.putText(colorimage, originstring, (x+w/2, y+h/2), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255))



                    Pose = np.mat([[storerotationvector[0, 0], storerotationvector[0, 1], storerotationvector[0, 1], P_c[0]], \
                                   [storerotationvector[0, 1], storerotationvector[1, 1], storerotationvector[1, 2], P_c[1]], \
                                   [storerotationvector[2, 0], storerotationvector[2, 1], storerotationvector[2, 2], P_c[2]], \
                                   [0, 0, 0, 1]])

                    Pose = np.mat([[1, 0, 0, P_c[0]], \
                                   [0, 1, 0, P_c[1]], \
                                   [0, 0, 1, P_c[2]], \
                                   [0, 0, 0, 1]])
                    
                    Inv_Pose = np.linalg.inv(Pose)

                    P_r = Inv_Pose * P_h

                    print(P_r)

                    #P_f = Inv_Pose * P_c


                    cv2.circle(colorimage, (960,510), 10, (0,0,255), 5)

        
        if(state == "haardetection"):
            range_min = cv2.getTrackbarPos('depthcamerarangemax','Haar Detection')
            range_max = cv2.getTrackbarPos('depthcamerarangemin','Haar Detection')
            scalefactor = cv2.getTrackbarPos('scalingfactor','Haar Detection')
            minneighbors = cv2.getTrackbarPos('minimumneighbors','Haar Detection')
            if(scalefactor < 2):
                scalefactor = 2

            deptharray = camera.getarray_depth()
            outputimage = camera.getimage_depth()

            detectoroutput = multiscaledetection(range_min, range_max, scalefactor, detector, minneighbors, deptharray)

            for (x,y,w,h) in detectoroutput:
                cv2.rectangle(outputimage,(x,y0),(x+w,y+h),(255,0,0),2)
                detecteddistance = deptharray[x+w/2, y+h/2]
                #print(detecteddistance)

                detectedpositonvector = projection.calculate([x+w/2, y+h/2, detecteddistance])

                #Show Location on image
                if(detecteddistance != 0):
                    originstring = str(detectedpositonvector[0]) + ", " + str(detectedpositonvector[1]) + ", " + str(detectedpositonvector[2])
                else:
                    originstring = "Too close to camera"
                
                cv2.putText(colorimage, originstring, (x+w/2, y+h/2), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255))




            
        #Show Images as per bool variables
        if(state == "depth"):
            cv2.imshow('Depth Image',depthimage)

        if(state == "imagecolection"):
            cv2.imshow('GreyScale Image Collection',depthimage)
            cv2.imshow('Crop',crop)
   
        if(state =="color"):
            cv2.imshow('Color Image',colorimage)

        if(state =="calibration"):
            cv2.imshow('Calibration',colorimage)

        if(state == "imagecolection"):
            cv2.imshow('Color Isolation Image',depthimage)

        if(state == "positiondetection"):
            cv2.imshow('Position Detection',colorimage)

        if(state == "posedetection"):
            cv2.imshow('Pose Detection',colorimage)

        if(state == "colorcalibration"):
            cv2.imshow('Color Calibration', colorimage)

        if(state == "haardetection"):
            cv2.imshow("Haar Detection", outputimage)
            try:
                cv2.imshow('Crop',outputcropimage)
            except:
                pass
            
        k = cv2.waitKey(1) & 0xFF

        if k == 100: # d for depth
            cv2.destroyAllWindows()
            cv2.namedWindow('Depth Image')
            cv2.createTrackbar('depthcamerarangemax','Depth Image',500,3000,nothing)
            cv2.createTrackbar('depthcamerarangemin','Depth Image',1000,3000,nothing)
            state = "depth"
            print("Depth Image Mode")
            print(" ")

        elif k == 99: # c for color
            cv2.destroyAllWindows() 
            state = "color"
            print("Color Image Mode")
            print(" ")

        elif k == 107: # K for Kolor Calibration
            cv2.destroyAllWindows() 
            state = "colorcalibration"
            mouseclick = []
            print("Color Calibration")
            print(" ")

        elif k == 97: # a for Calibration
            cv2.destroyAllWindows() 
            state = "calibration"
            print("Calibration")
            print(" ")

        elif k == 48: # d for position detection
            cv2.destroyAllWindows() 
            state = "positiondetection"
            cv2.namedWindow('Position Detection')
            cv2.createTrackbar('scalingfactor','Position Detection',1,100,nothing)
            cv2.setTrackbarPos('scalingfactor','Position Detection', 3)
            cv2.createTrackbar('minimumneighbors','Position Detection',1,50,nothing)
            cv2.setTrackbarPos('minimumneighbors','Position Detection', 30)
            cv2.createTrackbar('depthcamerarangemax','Position Detection',500,3000,nothing)
            cv2.createTrackbar('depthcamerarangemin','Position Detection',1000,3000,nothing)
            detector = cv2.CascadeClassifier("matlabhand4.xml")
            print("Position Detection")
            print(" ")            

        elif k == 112: # p for pose
            cv2.destroyAllWindows()
            state = "posedetection"
            print("Pose Detection Mode")
            print(" ")

        elif k == 105: # i for imagecollection
            cv2.destroyAllWindows()
            state = "imagecolection"
            print("Image Collection Mode")
            print(" ")


        elif k == 104: # h for haar
            cv2.destroyAllWindows()
            cv2.namedWindow('Haar Detection')
            cv2.createTrackbar('scalingfactor','Haar Detection',1,100,nothing)
            cv2.setTrackbarPos('scalingfactor','Haar Detection', 3)
            cv2.createTrackbar('minimumneighbors','Haar Detection',1,50,nothing)
            cv2.setTrackbarPos('minimumneighbors','Haar Detection', 30)
            cv2.createTrackbar('depthcamerarangemax','Haar Detection',500,3000,nothing)
            cv2.createTrackbar('depthcamerarangemin','Haar Detection',1000,3000,nothing)
            state = "haardetection"
            detector = cv2.CascadeClassifier("matlabhand4.xml")
            print("Haar Detection Mode")


        # elif k == 104: # 
        #     cv2.destroyAllWindows() 
        #     state = "depthhaar"
        #     print("Depth Mode Image Aquisition for haar detection")


        elif k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
            print("Exiting")
            return True

def nothing(num):
    pass

            


main()

