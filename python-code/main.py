import sys
from utils import *
from init_camera import initialize
from register_camera import register
import seaborn as sns


def init():
    ################################################
    # initialize configuration and camera dictionary
    ################################################
    k1=0
    k2=0
    path='../matlab-code/data/k/K.txt'
    f=open(path,'r')
    lines=f.readlines()
    K=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            K[i,j]=float(lines[i].split(' ')[j])

    DIR_IMAGES='../matlab-code/data/images/*.png'
    config=SFM_Params(DIR_IMAGES,hessianThreshold=1000,detector='SIFT',matcher='FLANN')
    dict_cameras={}
    n_cameras=config.n_cameras
    for i in range(n_cameras):
        dict_cameras[i]={}
    
    config.K=K
    config.DISTCOEFFS=np.array([k1,k2,0,0]).astype('float32')
    
    config.ratio_test_threshold=0.65
    config.pnp_threshold = 5
    config.dist_threshold=100
    config.post_threshold=5
    config.pre_threshold=24
    config.indice_registered_cameras=[]
    
    return config,dict_cameras

if __name__=='__main__':
    SAVE_NAME='fountain'
    if(not os.path.exists('results/%s' % SAVE_NAME)):
        os.makedirs('results/%s' % SAVE_NAME)
    ########################################################
    # init configurations 
    ########################################################
    config,dict_cameras=init()



    ########################################################
    # load images
    ########################################################
    dict_cameras=load_images(config,dict_cameras)



    ########################################################
    # extract features
    ########################################################
    dict_cameras=extract_features(config,dict_cameras)



    ########################################################
    # match features
    ########################################################
    dict_cameras=match_features(config,dict_cameras)
    ax = sns.heatmap(config.n_good_matches)



    ########################################################
    # initialize the first two images
    ########################################################
    config,dict_cameras=initialize(config,dict_cameras)
    np.savetxt('results/fountain/2.txt',config.reconstructed_points_3d,delimiter=';')



    ########################################################
    # register more cameras
    ########################################################
    n_cameras=config.n_cameras
    for i in range(n_cameras-2):
        print('-----------------------------------')
        print('Registering %dth camera............' % (i+3))
        config,dict_cameras,n_observations,n_points_3d=register(config,dict_cameras)
        print('%d observations, %d 3D points' %(n_observations,n_points_3d))

        mask=[]
        for crn_camera_index in config.indice_registered_cameras:
            mask.extend(dict_cameras[crn_camera_index]['point_indice'].tolist())
        mask=list(set(mask))
        reconstructed_points_3d=config.reconstructed_points_3d[mask]
        np.savetxt('results/fountain/%d.txt' % (i+3),reconstructed_points_3d,delimiter=';')