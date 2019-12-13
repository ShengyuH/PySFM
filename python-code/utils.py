# SURF and SIFT is only supported in older version
# pip install opencv-contrib-python==3.4.2.17
# pip install opencv-python==3.4.2.17
import numpy as np
import glob,os,sys,time
import cv2 as cv 
from scipy.spatial.transform import Rotation as R

class SFM_Params:
    def __init__(self,img_dir,n_images=None,maxFeatures=None,hessianThreshold=200,detector='SIFT',matcher='FLANN'):
        self.img_dir=img_dir
        files=sorted(glob.glob(self.img_dir))
        if(n_images is not None):
            self.files=files[:n_images]
        else:
            self.files=files
        self.n_cameras=len(self.files) # number of cameras
        self.n_good_matches=np.zeros((self.n_cameras,self.n_cameras))  # number of good matches between each camera
        
        # feature detectors
        if(maxFeatures is not None):
            sift = cv.xfeatures2d.SIFT_create(nfeatures=maxFeatures)
        else:
            sift = cv.xfeatures2d.SIFT_create()            
        surf = cv.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold,extended=True)   # surf detector   
        if(detector=='SIFT'):
            self.detector=sift
        elif(detector=='SURF'):
            self.detector=surf


        # feature matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE)
        search_params = dict()   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        bf = cv.BFMatcher()  # Brute Force knn matcher

        if(matcher=='FLANN'):
            self.matcher=flann
        elif(matcher=='BF'):
            self.matcher=bf

def read_image(path,K,distcoeffs=None,undistort=False,ratio=1):
    '''
    load images as gray images
    Input: 
        path:       image path
        K:          calibration matrix
        distcoeffs: distortion coefficient
    Output:
        img:        loaded 2D array
    '''
    img=cv.imread(path,0)
    if(undistort):
        img=cv.undistort(img,K,distcoeffs)
    return img

def load_images(config,dict_cameras):
    '''
    read all the images
    '''
    for i in range(config.n_cameras):
        img_path=config.files[i]
        img=read_image(img_path,config.K,distcoeffs=None,undistort=False)
        dict_cameras[i]['img']=img
    print('Loaded in total %d frames ' % config.n_cameras)
    return dict_cameras

def extract_features(config,dict_cameras):
    '''
    extract features
    '''
    t0=time.time()
    for i in range(config.n_cameras):
        img=dict_cameras[i]['img']
        kp, des = config.detector.detectAndCompute(img,None)
        dict_cameras[i]['kp'],dict_cameras[i]['des']=kp,des
    t1=time.time()
    print('Feature detection takes %d seconds' % (t1-t0))
    return dict_cameras

def match_features(config,dict_cameras):
    '''
    exhausitively match features
    '''
    t0=time.time()
    for i in range(config.n_cameras):
        dict_cameras[i]['matches']=[] # matches is a list
        des1=dict_cameras[i]['des']  # descriptors of the first camera, queryIdx
        for j in range(config.n_cameras):
            if(i!=j):
                des2=dict_cameras[j]['des']  # des1——query index; des2 —— train index
                matches = config.matcher.knnMatch(des1,des2,k=2)# find the best and second best match
                matches=sorted(matches,key=lambda x:x[0].distance) # sort by the distance first, ascending order
                good_match = []
                ind_train=[] # filter the cases when multiple queryIdx point to the same trainIdx
                for m,n in matches:
                    if m.distance < config.ratio_test_threshold*n.distance and m.trainIdx not in ind_train:
                        good_match.append(m)
                        ind_train.append(m.trainIdx)
                config.n_good_matches[i,j]=len(good_match)
                dict_cameras[i]['matches'].append(good_match)
            else:
                dict_cameras[i]['matches'].append([])
    t1=time.time()
    print('Feature matching takes %d seconds' % (t1-t0))
    return dict_cameras


def reprojection_error(points_3d,points_2d,P,K):
    '''
    calculate the reprojection error
    Input:
        points_3d:  4 x N
        points_2d:  N X 3
        P:          4 X 4
        K:          3 X 3
    Output:
        error:      N x 2
    '''
    reprojected_2d=np.dot(P,points_3d)[:3,:].T
    reprojected_2d=reprojected_2d[:,:2]/reprojected_2d[:,2,np.newaxis]
    error=reprojected_2d-points_2d[:,:2]
    error=np.dot(K[:2,:2],error.T).T
    return error


def linearTriangulation(P1,x1s,P2,x2s,K):
    '''
    Given two projection matrice and calibrated image points, triangulate them to get 3-D points
    and also get the reprojection error [pixel]
    Input:
        P1:     4 x 4
        P2:     4 x 4
        x1s:    N x 3
        x2s:    N x 3
        K:      3 x 3 
    Output:
        XS:     4 x N
        error:  N x 4
    '''
    XS=np.zeros((4,x1s.shape[0]))
    for k in range(x1s.shape[0]):
        r1=x1s[k,0]*P1[2,:]-P1[0,:]
        r2=x1s[k,1]*P1[2,:]-P1[1,:]
        r3=x2s[k,0]*P2[2,:]-P2[0,:]
        r4=x2s[k,1]*P2[2,:]-P2[1,:]
        
        A=np.vstack((r1,r2,r3,r4))
        _,_,Vh=np.linalg.svd(A)
        XS[:,k]=Vh.T[:,-1]/Vh.T[3,3]
    
    # get the reprojection errors
    error_1=reprojection_error(XS,x1s,P1,K)
    error_2=reprojection_error(XS,x2s,P2,K)
    
    error=np.hstack((error_1,error_2))
    return XS,error


def decomposeE(E,x1s,x2s,K):
    '''
    Given the essential and calibrated image points, get the second projection matrix and the index of inliers
    Input:
        E:      3 x 3
        x1s:    N x 3
        x2s:    N x 3
        K:      3 x 3

    Output:
        proj_mat:   4 x 4 
        ind_inlier: N x 1
    '''
    W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    [U,S,Vh]=np.linalg.svd(E) # U and Vh is normalized
    
    # extract translation
    t=U[:,2].reshape((-1,1))
    
    # extraction rotation
    R1=np.dot(np.dot(U,W),Vh)
    R2=np.dot(np.dot(U,W.T),Vh)
    if(np.linalg.det(R1)<0):
        R1=-R1
    if(np.linalg.det(R2)<0):
        R2=-R2
    
    # four possible projection matrice
    P1=np.vstack((np.hstack((R1,t)),np.array([0,0,0,1])))
    P2=np.vstack((np.hstack((R1,-t)),np.array([0,0,0,1])))
    P3=np.vstack((np.hstack((R2,t)),np.array([0,0,0,1])))
    P4=np.vstack((np.hstack((R2,-t)),np.array([0,0,0,1])))
    Ps=[P1,P2,P3,P4]
    
    # determine the projection matrix by the maximum inliers
    n_inliers=[]
    indice_inliers=[]
    P=np.eye(4) # the first projection matrix is identical
    for proj_mat in Ps:
        X,_=linearTriangulation(P,x1s,proj_mat,x2s,K)
        p1X=np.dot(P,X)
        p2X=np.dot(proj_mat,X)
        
        indice_inlier=np.logical_and(p1X[2,:]>0,p2X[2,:]>0)
        indice_inliers.append(indice_inlier)
        n_inliers.append(indice_inlier.sum())
    
    # determine the best projection matrix by the most reconstructed points in front of the cameras
    n_inliers=np.array(n_inliers)
    index_proj_mat=n_inliers.argmax() 
    
    ind_inlier=indice_inliers[index_proj_mat]
    proj_mat=Ps[index_proj_mat]
    return proj_mat,ind_inlier

def proj_mat_to_camera_vec(proj_mat):
    '''
    decompose the projection matrix to camera paras(rotation vector and translation vector)
    Input: 
        proj_mat:       4 x 4
    Output:
        camera_vec:     1 x 6
    
    '''
    rot_mat=proj_mat[:3,:3]
    r=R.from_dcm(rot_mat)
    rot_vec=r.as_rotvec()
    t_vec=proj_mat[:3,3]
    camera_vec=np.hstack((rot_vec,t_vec))
    return camera_vec

def recover_projection_matrix(camera_param):
    '''
    given camera parameters, recover the projection matrix
    Input:
        camera_param:   1 x 6
    Output:
        P:              4 x 4
        
    '''
    rot_vec=camera_param[:3]
    translate_vec=camera_param[3:]
    r=R.from_rotvec(rot_vec)
    rot_matrix=r.as_dcm()
    P=np.eye(4)
    P[:3,:3]=rot_matrix
    P[:3,3]=translate_vec.T
    return P