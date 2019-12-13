import numpy as np
import glob,os,sys,time
import cv2 as cv 
import matplotlib.pyplot as plt
import json
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import numpy.matlib
from collections import Counter
from utils import *
from bundle_adjustment import bundle_adjustment_sparsity,fun

def next_camera(config,dict_cameras):
    '''
    determine the next camera,return the camera pairs and # linked points
    '''
    max_overlapping=0
    index_camera_3d=0 # camera used to provide 3d points
    index_camera_2d=0 # camera  used to provide 2d points
    for j in config.indice_registered_cameras:   # find the registered cameras with the maximum overlap points
        crn_matches=dict_cameras[j]['matches'] # current matches with all the other cameras 
        indice_query_registered=dict_cameras[j]['indice_registered_2d'] # indice of reconstructed points
        for ii in range(config.n_cameras):
            if(ii not in config.indice_registered_cameras): # only inspect non-registered cameras
                crn_match=crn_matches[ii]
                indice_overlap=[ele for ele in crn_match if ele.queryIdx in indice_query_registered]
                n_overlap=len(indice_overlap)
                # update the maximum overlap
                if(n_overlap>max_overlapping):
                    max_overlapping=n_overlap
                    index_camera_3d=j
                    index_camera_2d=ii
    return index_camera_2d,index_camera_3d,max_overlapping

def PnP(config,crn_match,camera_2d,camera_3d):
    '''
    ransacP3P and return the new camera parameter, corresponding 2d and 3d indice
    '''
    indice_points_2d,indice_points_3d=[],[]
    new_points_2d=[]
    for ele in crn_match:
        check_exist=np.where(camera_3d['indice_registered_2d']==ele.queryIdx)[0]
        if(len(check_exist)!=0):
            assert len(check_exist)==1
            indice_points_2d.append(ele.trainIdx)
            new_points_2d.append(camera_2d['kp'][ele.trainIdx].pt)
        
            ind_3d=check_exist[0]
            indice_points_3d.append(camera_3d['point_indice'][ind_3d])
            
    new_points_2d=np.float64(new_points_2d)   
    existing_points_3d=config.reconstructed_points_3d[indice_points_3d]
    
    # set reprojection error threshold
    _,rvec,tvec,ind_inliers=cv.solvePnPRansac(objectPoints = existing_points_3d, imagePoints = new_points_2d, 
                                              cameraMatrix = config.K, distCoeffs = config.DISTCOEFFS,reprojectionError=config.pnp_threshold)
    camera_params=np.hstack((rvec.flatten(),tvec.flatten()))

    indice_points_2d=np.array(indice_points_2d)[ind_inliers]
    indice_points_3d=np.array(indice_points_3d)[ind_inliers]

    return camera_params,indice_points_2d,indice_points_3d

def reconstruct_3d_and_link_2d(config,crn_match,P1,P2,camera_new,camera_exist):
    '''
    reconstruct 2D points with all the other images
    # 1. if ele.queryIdx in camera_exist['indice_registered_2d']
    #    then it has been linked to a 3D point,check if ele.trainIdx is also registered
    # 2. if ele.queryIdx not in ... and ele.trainIdx not in
    #    then reconstruct a new 3D points
    # 3. if ele.queryIdx not in ... but ele.trainIdx in 
    #    then it should be linked to that 3D point, and this point 
    #    could be the 3D-2D one, or the new one, which is in case 2
    '''
    #####################################################################
    # add new observation
    ####################################################################
    pts1,pts2=[],[] # pts1 =====> camera_new =====> trainIdx
    indice_1,indice_2=[],[]
    indice_2d_exist,indice_point_exist=[],[] # used for case 3
    indice_2d_new,indice_point_new=[],[] # used for case 1
    for ele in crn_match:
        check_query=np.where(camera_exist['indice_registered_2d']==ele.queryIdx)[0]
        check_train=np.where(camera_new['indice_registered_2d']==ele.trainIdx)[0]
        # case 1, link the observation
        if(len(check_query)==1):
            if(len(check_train)==0):
                indice_2d_new.append(ele.trainIdx)
                crn_point_index=camera_exist['point_indice'][check_query[0]]
                indice_point_new.append(crn_point_index)          
        else:
            # case 2, reconstruct new one
            if(len(check_train)==0):
                pts1.append(camera_new['kp'][ele.trainIdx].pt)
                pts2.append(camera_exist['kp'][ele.queryIdx].pt)
                indice_1.append(ele.trainIdx)
                indice_2.append(ele.queryIdx)
            # case 3, link the observation
            else:
                crn_point_index=camera_new['point_indice'][check_train[0]]                    
                indice_2d_exist.append(ele.queryIdx)
                indice_point_exist.append(crn_point_index)
        
                
    #####################################################################
    # filter outliers before BA
    ####################################################################
    if(len(pts1)!=0):
        pts1=np.float64(pts1)
        pts2=np.float64(pts2)
        indice_1=np.array(indice_1)
        indice_2=np.array(indice_2)

        # calibrate the homogeneous points and reconstruct them
        if(pts1.shape[1]==2):
            pts1=np.hstack((pts1,np.ones((pts1.shape[0],1))))
        if(pts2.shape[1]==2):
            pts2=np.hstack((pts2,np.ones((pts2.shape[0],1))))
        
        pts1=np.dot(np.linalg.inv(config.K),pts1.T).T
        pts2=np.dot(np.linalg.inv(config.K),pts2.T).T
        XS,error=linearTriangulation(P1,pts1,P2,pts2,config.K)
        
        ii_inlier=(np.abs(error)<config.pre_threshold).sum(1)==4  # filter by reprojection error
        jj_inlier=np.linalg.norm(XS.T[:,:3]-config.ORIGIN,axis=1)<config.dist_threshold   # filter by the distance to the origin
        mask=np.logical_and(ii_inlier,jj_inlier)
        XS=XS[:,mask]
        indice_1=indice_1[mask]
        indice_2=indice_2[mask]

        # update reconstructed 3D points
        new_points_3d=XS[:3,:].T
        n_new_points_3d=new_points_3d.shape[0]
        n_exist_points_3d=config.reconstructed_points_3d.shape[0]
        new_point_indice=np.arange(n_exist_points_3d,n_exist_points_3d+n_new_points_3d)
        tmp=config.reconstructed_points_3d
        config.reconstructed_points_3d=np.vstack((tmp,new_points_3d))

        # update the new camera
        tmp=camera_new['indice_registered_2d']
        camera_new['indice_registered_2d']=np.hstack((tmp,indice_1))
        tmp=camera_new['point_indice']
        camera_new['point_indice']=np.hstack((tmp,new_point_indice))

        # update the existing camera
        tmp=camera_exist['indice_registered_2d']
        camera_exist['indice_registered_2d']=np.hstack((tmp,indice_2))
        tmp=camera_exist['point_indice']
        camera_exist['point_indice']=np.hstack((tmp,new_point_indice))
        
    # filter newly linked observations with big reprojection errors 
    if(len(indice_point_exist)!=0):
        points_3d=config.reconstructed_points_3d[indice_point_exist]
        points_3d=np.hstack((points_3d,np.ones((points_3d.shape[0],1))))
        proj_mat=recover_projection_matrix(camera_exist['camera'])

        points_2d=[camera_exist['kp'][ele].pt for ele in indice_2d_exist]
        points_2d=np.float64(points_2d)
        points_2d=np.hstack((points_2d,np.ones((points_2d.shape[0],1))))
        points_2d=np.dot(np.linalg.inv(config.K),points_2d.T).T[:,:2]

        error=reprojection_error(points_3d.T,points_2d,proj_mat,config.K)

        mask=(np.abs(error)<config.pre_threshold).sum(1)==2
        indice_point_exist=np.array(indice_point_exist)[mask]
        indice_2d_exist=np.array(indice_2d_exist)[mask]

        if(indice_point_exist.shape[0]!=0):
            tmp=camera_exist['point_indice']
            camera_exist['point_indice']=np.hstack((tmp,indice_point_exist))
            tmp=camera_exist['indice_registered_2d']
            camera_exist['indice_registered_2d']=np.hstack((tmp,indice_2d_exist))

    if(len(indice_2d_new)!=0):
        points_3d=config.reconstructed_points_3d[indice_point_new]
        points_3d=np.hstack((points_3d,np.ones((points_3d.shape[0],1))))
        proj_mat=recover_projection_matrix(camera_new['camera'])
        points_2d=[camera_new['kp'][ele].pt for ele in indice_2d_new]
        points_2d=np.float64(points_2d)
        points_2d=np.hstack((points_2d,np.ones((points_2d.shape[0],1))))
        points_2d=np.dot(np.linalg.inv(config.K),points_2d.T).T[:,:2]

        error=reprojection_error(points_3d.T,points_2d,proj_mat,config.K)
        mask=(np.abs(error)<config.pre_threshold).sum(1)==2
        indice_point_new=np.array(indice_point_new)[mask]
        indice_2d_new=np.array(indice_2d_new)[mask]

        if(indice_2d_new.shape[0]!=0):
            tmp=camera_new['point_indice']
            camera_new['point_indice']=np.hstack((tmp,indice_point_new))
            tmp=camera_new['indice_registered_2d']
            camera_new['indice_registered_2d']=np.hstack((tmp,indice_2d_new))
        

    return camera_new,camera_exist,config


def register(config,dict_cameras):
    #############################################################################
    # determine the next camera to be registered and linked
    #############################################################################
    index_camera_2d,index_camera_3d,n_pairs=next_camera(config,dict_cameras)
    
    ##############################################################################
    # ransacPnP
    ##############################################################################
    crn_match=dict_cameras[index_camera_3d]['matches'][index_camera_2d] # match from 3D to 2D
    camera_3d=dict_cameras[index_camera_3d]  
    camera_2d=dict_cameras[index_camera_2d]
    camera_param,indice_points_2d,indice_points_3d =PnP(config,crn_match,camera_2d,camera_3d)
    
    # update the parameters of new camera
    dict_cameras[index_camera_2d]['camera']=camera_param
    dict_cameras[index_camera_2d]['indice_registered_2d']=indice_points_2d.flatten()
    dict_cameras[index_camera_2d]['point_indice']=indice_points_3d.flatten()

    print('Use %d out of %d points to register camera %d by camera %d' % (len(indice_points_2d),n_pairs,index_camera_2d,index_camera_3d))
    
    ##################################################################################
    # reconstruct 3d points AND link new observations with all the other images
    ##################################################################################
    P1=recover_projection_matrix(camera_param)
    tmp=config.indice_registered_cameras
    for index_camera in tmp:
        crn_match=dict_cameras[index_camera]['matches'][index_camera_2d]
        P2=recover_projection_matrix(dict_cameras[index_camera]['camera'])
        camera_new=dict_cameras[index_camera_2d]
        camera_exist=dict_cameras[index_camera]
        dict_cameras[index_camera_2d],dict_cameras[index_camera],config=reconstruct_3d_and_link_2d(config,crn_match,P1,P2,camera_new,camera_exist)
            
    ######################################################
    # run BA
    ######################################################
    config.indice_registered_cameras.append(index_camera_2d)
    n_cameras=len(config.indice_registered_cameras)
    n_points=config.reconstructed_points_3d.shape[0]
    
    # points_3d, n_points x 3
    points_3d=config.reconstructed_points_3d
    
    # get camera params,points_2d, camera_indice
    camera_params=np.zeros((n_cameras,6))
    camera_indices=[]
    point_indices=[]
    points_2d=[]
    for j in range(n_cameras):
        crn_camera_index=config.indice_registered_cameras[j]
        
        camera_params[j,:]=dict_cameras[crn_camera_index]['camera']
        camera_indices.extend([j]*dict_cameras[crn_camera_index]['indice_registered_2d'].shape[0])
        point_indices.extend(dict_cameras[crn_camera_index]['point_indice'].tolist())
        
        for ele in dict_cameras[crn_camera_index]['indice_registered_2d']:
            points_2d.append(dict_cameras[crn_camera_index]['kp'][ele].pt)
            
    points_2d=np.float64(points_2d)
    camera_indices=np.array(camera_indices)  
    point_indices=np.array(point_indices)  
    
    # calibrate the 2d image pts to calculate residual
    if(points_2d.shape[1]==2):
        points_2d=np.hstack((points_2d,np.ones((points_2d.shape[0],1))))
    points_2d=np.dot(np.linalg.inv(config.K),points_2d.T).T
    points_2d=points_2d[:,:2]
    
    # optimize
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(fun, x0,jac='3-point',jac_sparsity=A, verbose=1, x_scale='jac', ftol=1e-5, method='trf',loss='soft_l1',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d,config.K))
    
    ###################################################################################
    # filter the big outliers and distant 3D points by updateing the camera dictionary
    ###################################################################################
    optimized_params=res.x   
    camera_params=optimized_params[:n_cameras*6].reshape((n_cameras,6))
    points_3d=optimized_params[n_cameras*6:].reshape((n_points,3))
    
    # filter by reprojection error
    ff=np.abs(res.fun.reshape((-1,2)))
    ii_inlier=(ff<config.post_threshold).sum(1)==2
    
    # filter by the distance to the origin
    pts_3d=points_3d[point_indices]
    jj_inlier=np.linalg.norm(pts_3d-config.ORIGIN,axis=1)<config.dist_threshold
    
    mask=np.logical_and(ii_inlier,jj_inlier)
    
    # filter if there's only 1 support FOR 3D point
    left_point_indices=point_indices[mask]
    count_support=dict(Counter(left_point_indices.tolist()))
    remove_indice=[]
    for key,value in count_support.items():
        if(value==1):
            remove_indice.append(key)
    ii_mask=[True if ele not in remove_indice else False for ele in point_indices]
    mask=np.logical_and(mask,ii_mask)
    

    # update camera dictionary
    n_observations=mask.sum()   
    n_points_3d=len(list(set(point_indices[mask]))) 
    count=0
    for j in range(n_cameras):
        crn_camera_index=config.indice_registered_cameras[j]
        n_eles=dict_cameras[crn_camera_index]['indice_registered_2d'].shape[0]
        ii_mask=mask[count:count+n_eles]
        count+=n_eles
        dict_cameras[crn_camera_index]['indice_registered_2d']=dict_cameras[crn_camera_index]['indice_registered_2d'][ii_mask]
        dict_cameras[crn_camera_index]['point_indice']=dict_cameras[crn_camera_index]['point_indice'][ii_mask]           
        dict_cameras[crn_camera_index]['camera']=camera_params[j,:]
    config.reconstructed_points_3d=points_3d

    return config,dict_cameras,n_observations,n_points_3d