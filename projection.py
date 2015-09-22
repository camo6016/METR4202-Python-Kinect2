import math
from numpy import *
import numpy as np
from numpy.linalg import norm
import cv2

class translator(object):
    def __init__(self, focallength, imagecenter, camerawidth, cameraheight):
        self.focallength = focallength
        self.height = cameraheight
        self.width = camerawidth
        self.imagecenter = imagecenter

    def calculate(self, vector):
        #print(vector)

        x_in = vector[0]
        y_in = vector[1]
        z_in = vector[2]

        if(z_in==0):
            return 0.0,0.0,0.0

        in_matrix = mat([[x_in],[y_in],[1]])
        matrix = mat([[self.focallength[0],0,self.imagecenter[0],0],[0,self.focallength[1],self.imagecenter[1],0],[0,0,1,0]])
        inv_matrix = linalg.pinv(matrix)
        out_matrix = inv_matrix * in_matrix

        z_ratio = sqrt((out_matrix[0]*out_matrix[0]) + (out_matrix[1]*out_matrix[1]) + (out_matrix[2]*out_matrix[2]))
        out_matrix = out_matrix * (z_in/float(z_ratio[0,0]))

        return [round(out_matrix[0],1), -round(out_matrix[1],1), round(out_matrix[2],1)]

class pose(object):
    def __init__(self):
        self.foundpose = False
        self.checkerboardwidth = 4
        self.checkerboardheight = 3
        numberofsqueres = self.checkerboardwidth * self.checkerboardheight
        self.criterias = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.005)
        self.objp = np.zeros((numberofsqueres,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.checkerboardwidth,0:self.checkerboardheight].T.reshape(-1,2)
        
        self.axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    def findpose(self, colorimage, deptharray, mtx, dist):
        gray = cv2.cvtColor(colorimage,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.checkerboardwidth,self.checkerboardheight), None, 11)
        self.corners = corners

        if(ret==False):
            return False, False, False, False, False

        retval, rvecs, tvecs = cv2.solvePnP(self.objp, self.corners, mtx, dist)

        rotationvector = rpy2r(rvecs[0][0],rvecs[1][0],rvecs[2][0])

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(self.axis, rvecs, tvecs, mtx, dist)
        
        corner = tuple(self.corners[0].ravel())

        origin_distance = deptharray[corner[1], corner[0]]

        outputvector = [corner[0],corner[1],origin_distance]

        return outputvector, rotationvector, True, corners, imgpts

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img



def rotx(theta):
    """
    Rotation about X-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about X-axis

    @see: L{roty}, L{rotz}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)
    return mat([[1,  0,    0],
            [0,  ct, -st],
            [0,  st,  ct]])

def roty(theta):
    """
    Rotation about Y-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Y-axis

    @see: L{rotx}, L{rotz}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)

    return mat([[ct,   0,   st],
            [0,    1,    0],
            [-st,  0,   ct]])

def rotz(theta):
    """
    Rotation about Z-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Z-axis

    @see: L{rotx}, L{roty}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)

    return mat([[ct,      -st,  0],
            [st,       ct,  0],
            [ 0,    0,  1]])


def rpy2r(roll, pitch=None,yaw=None):
    """
    Rotation from RPY angles.
    
    Two call forms:
        - R = rpy2r(S{theta}, S{phi}, S{psi})
        - R = rpy2r([S{theta}, S{phi}, S{psi}])
    These correspond to rotations about the Z, Y, X axes respectively.

    @type roll: number or list/array/matrix of angles
    @param roll: roll angle, or a list/array/matrix of angles
    @type pitch: number
    @param pitch: pitch angle
    @type yaw: number
    @param yaw: yaw angle
    @rtype: 4x4 homogenous matrix
    @return: R([S{theta} S{phi} S{psi}])

    @see:  L{tr2rpy}, L{rpy2r}, L{tr2eul}

    """
    n=1
    if pitch==None and yaw==None:
        roll= mat(roll)
        if numcols(roll) != 3:
            error('bad arguments')
        n = numrows(roll)
        pitch = roll[:,1]
        yaw = roll[:,2]
        roll = roll[:,0]
    if n>1:
        R = []
        for i in range(0,n):
            r = rotz(roll[i,0]) * roty(pitch[i,0]) * rotx(yaw[i,0])
            R.append(r)
        return R
    try:
        r = rotz(roll[0,0]) * roty(pitch[0,0]) * rotx(yaw[0,0])
        return r
    except:
        r = rotz(roll) * roty(pitch) * rotx(yaw)
        return r

