import numpy as np
import glob,os,sys
import cv2 as cv 
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import numpy.matlib
from utils import decomposeE,linearTriangulation,proj_mat_to_camera_vec
from bundle_adjustment import bundle_adjustment_sparsity,fun

def initialize(config,dict_cameras):
    ############################################################
    # determine initial pair
    ############################################################

    ind = np.unravel_index(np.argmax(config.n_good_matches, axis=None), config.n_good_matches.shape)
    # set current match
    ind_img1=ind[0]
    ind_img2=ind[1]

    c_match=dict_cameras[ind_img1]['matches'][ind_img2]
    # recover 2D pts
    pts1,pts2=[],[]
    for ele in c_match:
        pts1.append(dict_cameras[ind_img1]['kp'][ele.queryIdx].pt)
        pts2.append(dict_cameras[ind_img2]['kp'][ele.trainIdx].pt)    
    pts1=np.int32(pts1)
    pts2=np.int32(pts2)
    n_pairs=pts1.shape[0]

    # compute F
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC,1,0.99999)


    # filter the outliers 
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # calibrate the homogeneous points
    if(pts1.shape[1]==2):
        pts1=np.hstack((pts1,np.ones((pts1.shape[0],1))))
    if(pts2.shape[1]==2):
        pts2=np.hstack((pts2,np.ones((pts2.shape[0],1))))
    img1_calibrated_inliers=np.dot(np.linalg.inv(config.K),pts1.T).T
    img2_calibrated_inliers=np.dot(np.linalg.inv(config.K),pts2.T).T

    # get Essential matrix from calibration matrix and fundamental matrix
    E=np.dot(np.dot(config.K.T,F),config.K)

    projection_matrix, ind_inliers=decomposeE(E,img1_calibrated_inliers,img2_calibrated_inliers,config.K)


    #############################################################
    # bundle adjustment
    #############################################################
    P1=np.eye(4)
    P2=projection_matrix
    inliers1=img1_calibrated_inliers[ind_inliers]
    inliers2=img2_calibrated_inliers[ind_inliers]
    XS,_=linearTriangulation(P1,inliers1,P2,inliers2,config.K)

    n_cameras=2
    n_points=XS.shape[1]

    # get camera params,  n_cameras x 6
    camera_params=np.zeros((n_cameras,6))
    camera_params[0,:]=proj_mat_to_camera_vec(P1)
    camera_params[1,:]=proj_mat_to_camera_vec(P2)

    # points_3d, n_points x 3
    points_3d=XS[:3,:].T

    # get camera_indice
    camera_indices=[]
    for i in range(n_cameras):
        camera_indices.extend([i]*n_points)
    camera_indices=np.array(camera_indices)

    # get point_indice
    point_indices=[]
    for i in range(n_cameras):
        for j in range(n_points):
            point_indices.append(j)
    point_indices=np.array(point_indices)

    # get point_2d
    points_2d_1=inliers1[:,:2]
    points_2d_2=inliers2[:,:2]
    points_2d=np.vstack((points_2d_1,points_2d_2))

    # get params
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d,config.K)
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.plot(f0)

    # optimize
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(fun, x0,jac='3-point',jac_sparsity=A, verbose=1, x_scale='jac', ftol=1e-6, 
                        method='trf',loss='soft_l1',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d,config.K))
    plt.subplot(122)
    plt.plot(res.fun)

    #########################################
    # update the parameters
    #########################################
    config.indice_registered_cameras.append(ind_img1)
    config.indice_registered_cameras.append(ind_img2)

    # update the optimized parameters
    optimized_params=res.x
    camera_params=optimized_params[:n_cameras*6].reshape((n_cameras,6))
    points_3d=optimized_params[n_cameras*6:].reshape((n_points,3))

    # initial query and train indice
    query_indice=[ele.queryIdx for ele in c_match] # img1
    train_indice=[ele.trainIdx for ele in c_match] # img2

    # filter by ransacFundamental and decomposeE
    query_indice=np.array(query_indice)
    train_indice=np.array(train_indice)

    mask_F=mask.ravel()==1
    mask_E=ind_inliers
    query_indice=query_indice[mask_F][mask_E]
    train_indice=train_indice[mask_F][mask_E]

    # filter observations with big reprojection error
    ff=np.abs(res.fun.reshape((-1,2)))
    ii_inlier=(ff<config.post_threshold).sum(1)==2
    mask_query=ii_inlier[:query_indice.shape[0]]
    mask_train=ii_inlier[query_indice.shape[0]:]
    mask=np.logical_and(mask_query,mask_train)
    query_indice=query_indice[mask] 
    train_indice=train_indice[mask]
    config.reconstructed_points_3d=points_3d[mask]

    # filter outlier by distance to the origin
    ORIGIN=np.mean(config.reconstructed_points_3d,0)
    config.ORIGIN=ORIGIN
    distance=np.linalg.norm(config.reconstructed_points_3d-ORIGIN,axis=1)
    ii_inlier=distance<config.dist_threshold
    query_indice=query_indice[ii_inlier]
    train_indice=train_indice[ii_inlier]
    config.reconstructed_points_3d=config.reconstructed_points_3d[ii_inlier]
    print('Use camera %d and camera %d to initialize, %d pairs used, %d pairs reconstructed ' 
        % (ind_img1,ind_img2,n_pairs,ii_inlier.sum()))

    # update the camera dictionary
    dict_cameras[ind_img1]['camera']=camera_params[0]
    dict_cameras[ind_img1]['indice_registered_2d']=query_indice # indice of the reconstructed 2d points
    dict_cameras[ind_img1]['point_indice']=np.arange(len(query_indice)) # indice mapping to the 3D points
    dict_cameras[ind_img2]['camera']=camera_params[1]
    dict_cameras[ind_img2]['indice_registered_2d']=train_indice
    dict_cameras[ind_img2]['point_indice']=np.arange(len(train_indice))

    plt.figure()
    plt.plot(np.linalg.norm(config.reconstructed_points_3d-ORIGIN,axis=1))
    plt.title('distance to the origin')

    return config,dict_cameras

