__author__ = 'cxwang, mhgou'
__version__ = '1.0'

import os
import time
import numpy as np
import open3d as o3d
from transforms3d.euler import euler2mat, quat2mat

from ..grasp import GraspGroup
from .rotation import batch_viewpoint_params_to_matrix, matrix_to_dexnet_params

from .dexnet.grasping.quality import PointGraspMetrics3D
from .dexnet.grasping.grasp import ParallelJawPtGrasp3D 
from .dexnet.grasping.graspable_object import GraspableObject3D
from .dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
from .dexnet.grasping.contacts import Contact3D
from .dexnet.grasping.meshpy.obj_file import ObjFile
from .dexnet.grasping.meshpy.sdf_file import SdfFile

def get_scene_name(num):
    '''
    **Input:**
    - num: int of the scene number.
    
    **Output:**
    - string of the scene name.
    '''
    return ('scene_%04d' % (num,))

def create_table_points(lx, ly, lz, dx=0, dy=0, dz=0, grid_size=0.01):
    '''
    **Input:**
    - lx:
    - ly:
    - lz:
    **Output:**
    - numpy array of the points with shape (-1, 3).
    '''
    xmap = np.linspace(0, lx, int(lx/grid_size))
    ymap = np.linspace(0, ly, int(ly/grid_size))
    zmap = np.linspace(0, lz, int(lz/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, ymap, zmap], axis=-1)
    points = points.reshape([-1, 3])
    return points

def parse_posevector(posevector):
    '''
    **Input:**
    - posevector: list of pose
    **Output:**
    - obj_idx: int of the index of object.
    - mat: numpy array of shape (4, 4) of the 6D pose of object.
    '''
    mat = np.zeros([4,4],dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[:3,3] = posevector[1:4]
    mat[3,3] = 1
    obj_idx = int(posevector[0])
    return obj_idx, mat

def load_dexnet_model(data_path):
    '''
    **Input:**
        
    - data_path: path to load .obj & .sdf files
    
    **Output:**
    - obj: dexnet model
    '''
    of = ObjFile('{}.obj'.format(data_path))
    sf = SdfFile('{}.sdf'.format(data_path))
    mesh = of.read()
    sdf = sf.read()
    obj = GraspableObject3D(sdf, mesh)
    return obj

def transform_points(points, trans):
    '''
    **Input:**
    
    - points: (N, 3)
    
    - trans: (4, 4)
    
    **Output:**
    - points_trans: (N, 3)
    '''
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    points_trans = points_[:,:3]
    return points_trans

def compute_point_distance(A, B):
    '''
    **Input:**
    - A: (N, 3)
    
    - B: (M, 3)
    
    **Output:**
    - dists: (N, M)
    '''
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A-B, axis=-1)
    return dists

def compute_closest_points(A, B):
    '''
    **Input:**
    
    - A: (N, 3)
        
    - B: (M, 3)
    
    **Output:**
    
    - indices: (N,) closest point index in B for each point in A
    '''
    dists = compute_point_distance(A, B)
    indices = np.argmin(dists, axis=-1)
    return indices

def voxel_sample_points(points, voxel_size=0.008):
    '''
    **Input:**
    
    - points: (N, 3)
    
    **Output:**
    
    - points: (n, 3)
    '''
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points)
    return points

def topk_grasps(grasps, k=10):
    '''
    **Input:**
    
    - grasps: (N, 17)
    
    - k: int
    
    **Output:**
    
    - topk_grasps: (k, 17)
    '''
    assert(k > 0)
    grasp_confidence = grasps[:, 0]
    indices = np.argsort(-grasp_confidence)
    topk_indices = indices[:min(k, len(grasps))]
    topk_grasps = grasps[topk_indices]
    return topk_grasps

def get_grasp_score(grasp, obj, fc_list, force_closure_quality_config):
    """
    quality in this context is the minimum coefficient of friction required for the grasp to achieve force closure (i.e., a stable grasp).

    The function `get_grasp_score` tests the grasp at decreasing friction coefficients (from high to low).
    It returns the lowest friction coefficient at which the grasp is still stable. If the grasp fails at a certain friction,
    it returns the last successful (higher) friction value. If it never succeeds, it returns -1.

    So, a **lower score means a better grasp** (it requires less friction to be stable).

    example: fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    """
    tmp, is_force_closure = False, False
    quality = -1
    for ind_, value_fc in enumerate(fc_list):
        value_fc = round(value_fc, 2)
        tmp = is_force_closure
        is_force_closure = PointGraspMetrics3D.grasp_quality(grasp, obj, force_closure_quality_config[value_fc])
        if tmp and not is_force_closure: # if you fail at any value, return the last successful value
            quality = round(fc_list[ind_ - 1], 2)
            break
        elif is_force_closure and value_fc == fc_list[-1]: # if succed at 0.2, return 0.2 as nothing else to check
            quality = value_fc
            break
        elif value_fc == fc_list[0] and not is_force_closure: # if you fail at 1.2 then return -1
            break
    return quality

def collision_detection(grasp_list, model_list, dexnet_models, poses, scene_points, outlier=0.05, empty_thresh=10, return_dexgrasps=False):
    '''
    **Input:**
    
    - grasp_list: [(k1, 17), (k2, 17), ..., (kn, 17)] in camera coordinate
    
    - model_list: [(N1, 3), (N2, 3), ..., (Nn, 3)] in camera coordinate
    
    - dexnet_models: [GraspableObject3D,] in object coordinate
    
    - poses: [(4, 4),] from model coordinate to camera coordinate
    
    - scene_points: (Ns, 3) in camera coordinate
    
    - outlier: float, used to compute workspace mask
    
    - empty_thresh: int, 'num_inner_points < empty_thresh' means empty grasp

    - return_dexgrasps: bool, return grasps in dex-net format while True
    
    **Output:**
    
    - collsion_mask_list: [(k1,), (k2,), ..., (kn,)]

    - empty_mask_list: [(k1,), (k2,), ..., (kn,)]
    
    - dexgrasp_list: [[ParallelJawPtGrasp3D,],] in object coordinate
    '''
    # I guess labels were created with these values... so if I change then the labels no longer work!
    # These algorithims will learn to grasp with this gripper, that is fine... basically I don't really need to tune these values ever at all!

    # [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
    height = 0.02 # front to back, what we call finger_width, its hard coded here to be a size buffer! instead of reading from grasp list...
    depth_base = 0.02 # from the center of gripper to base (default 2cm), this is never passed
    finger_width = 0.01 # what we call finger_thickness, measured from inside to outside surface, this is never passed
    collision_mask_list = list()
    num_models = len(model_list)
    empty_mask_list = list()
    dexgrasp_list = list()

    for i in range(num_models):
        if len(grasp_list[i]) == 0:
            collision_mask_list.append(list())
            empty_mask_list.append(list())
            if return_dexgrasps:
                dexgrasp_list.append(list())
            continue

        ## parse grasp parameters
        # [score, width, height, depth, rotation_matrix(9), translation(3), object_id]

        model = model_list[i]
        obj_pose = poses[i]
        dexnet_model = dexnet_models[i]
        grasps = grasp_list[i]
        grasp_points = grasps[:, 13:16]
        grasp_poses = grasps[:, 4:13].reshape([-1,3,3])
        grasp_depths = grasps[:, 3]  # finger height - measured from center to finger tip
        grasp_widths = grasps[:, 1]  # grasp width

        ## crop scene, remove outlier
        xmin, xmax = model[:,0].min(), model[:,0].max()
        ymin, ymax = model[:,1].min(), model[:,1].max()
        zmin, zmax = model[:,2].min(), model[:,2].max()
        xlim = ((scene_points[:,0] > xmin-outlier) & (scene_points[:,0] < xmax+outlier))
        ylim = ((scene_points[:,1] > ymin-outlier) & (scene_points[:,1] < ymax+outlier))
        zlim = ((scene_points[:,2] > zmin-outlier) & (scene_points[:,2] < zmax+outlier))
        workspace = scene_points[xlim & ylim & zlim]
        
        # transform scene to gripper frame
        target = (workspace[np.newaxis,:,:] - grasp_points[:,np.newaxis,:])
        target = np.matmul(target, grasp_poses)
        
        # compute collision mask (vectorized for all grasps, very impressive!)
        mask1 = ((target[:,:,2]>-height/2) & (target[:,:,2]<height/2))
        mask2 = ((target[:,:,0]>-depth_base) & (target[:,:,0]<grasp_depths[:,np.newaxis]))
        mask3 = (target[:,:,1]>-(grasp_widths[:,np.newaxis]/2+finger_width))
        mask4 = (target[:,:,1]<-grasp_widths[:,np.newaxis]/2)
        mask5 = (target[:,:,1]<(grasp_widths[:,np.newaxis]/2+finger_width))
        mask6 = (target[:,:,1]>grasp_widths[:,np.newaxis]/2)
        mask7 = ((target[:,:,0]>-(depth_base+finger_width)) & (target[:,:,0]<-depth_base))
        
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        inner_mask = (mask1 & mask2 &(~mask4) & (~mask6))
        collision_mask = np.any((left_mask | right_mask | bottom_mask), axis=-1)
        empty_mask = (np.sum(inner_mask, axis=-1) < empty_thresh)
        collision_mask = (collision_mask | empty_mask)
        collision_mask_list.append(collision_mask)
        empty_mask_list.append(empty_mask)

        ## generate grasps in dex-net format
        if return_dexgrasps:
            dexgrasps = list()
            for grasp_id,_ in enumerate(grasps):
                grasp_point = grasp_points[grasp_id]
                R = grasp_poses[grasp_id]
                width = grasp_widths[grasp_id]
                depth = grasp_depths[grasp_id]
                points_in_gripper = target[grasp_id][inner_mask[grasp_id]]
                if empty_mask[grasp_id]:
                    dexgrasps.append(None)
                    continue
                center = np.array([depth, 0, 0]).reshape([3, 1]) # gripper coordinate
                center = np.dot(grasp_poses[grasp_id], center).reshape([3])
                center = (center + grasp_point).reshape([1,3]) # camera coordinate
                center = transform_points(center, np.linalg.inv(obj_pose)).reshape([3]) # object coordinate
                R = np.dot(obj_pose[:3,:3].T, R)
                binormal, approach_angle = matrix_to_dexnet_params(R)
                grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
                                            center, binormal, width, approach_angle), depth)
                dexgrasps.append(grasp)
            dexgrasp_list.append(dexgrasps)
    
    if return_dexgrasps:
        return collision_mask_list, empty_mask_list, dexgrasp_list
    else:
        return collision_mask_list, empty_mask_list

def eval_all_grasp(grasp_group: GraspGroup, models, dexnet_models, poses, config, table=None, voxel_size=0.008):
    """ Copy pasted and edited from eval_grasp() """
    num_models = len(models)

    #region - 1. Assign grasps to objects
    model_trans_list = list()
    seg_mask = list()
    for i,model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        # Transform model points to camera coordinates
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        # Create a mask labeling each point with its object index
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)
    # Concatenate all transformed model points into one scene point cloud

    # assign grasps
    # Compute closest points on scene (N, 3) and then look up object with (N, 1) mask
    indices = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[indices]
    #endregion - Assign grasps to objects

    #region - 2. Create grasp list
    grasp_list = list()
    orig_indices_list = list()  # <-- Add this

    for i in range(num_models):
        mask = (model_to_grasp == i)
        grasp_i = grasp_group[mask]
        grasp_list.append(grasp_i.grasp_group_array)
        orig_indices_list.append(np.where(mask)[0])


    #endregion - Create grasp list

    #region - 3. Using scene point cloud, check for collision and empty grasps
    if table is not None:  # If a table is provided, add its points to the scene (N + T,3)
        scene = np.concatenate([scene, table])

    # Check for collisions and empty grasps, and convert to Dex-Net format
    # grasps that are in the empty_list will be added to dexgrasp_list as None
    # which will skip below. You skip grasps that in in colission or empty
    collision_mask_list, empty_list, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_dexgrasps=True)
    #endregion

    #region - 4. Evaluate force closure quality of the grasps
    force_closure_quality_config = dict()
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])

    score_list = list()
    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        collision_mask = collision_mask_list[i]
        dexgrasps = dexgrasp_list[i]
        scores = list()
        num_grasps = len(dexgrasps)
        for grasp_id in range(num_grasps):
            if collision_mask[grasp_id]:
                scores.append(-1.)
                continue
            if dexgrasps[grasp_id] is None:
                scores.append(-1.)
                continue
            grasp = dexgrasps[grasp_id]
            score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)
            scores.append(score)
        score_list.append(np.array(scores))
    #endregion

    #region - 5. Prepare flat arrays in the original order
    flat_grasps = np.full((len(grasp_group), grasp_list[0].shape[1]), np.nan)
    flat_scores = np.full(len(grasp_group), np.nan)
    flat_collision = np.full(len(grasp_group), np.nan)

    for obj_idx, indices in enumerate(orig_indices_list):
        grasps = grasp_list[obj_idx]
        scores = score_list[obj_idx]
        collisions = collision_mask_list[obj_idx]
        for j, orig_idx in enumerate(indices):
            flat_grasps[orig_idx] = grasps[j]
            flat_scores[orig_idx] = scores[j]
            flat_collision[orig_idx] = collisions[j]
    # Check for any unfilled (nan) values
    assert not np.any(np.isnan(flat_grasps)), "flat_grasps contains NaN values, not all grasps copied over"
    #endregion - Prepare flat arrays in the original order

    # Now flat_grasps, flat_scores, and flat_collision are in the same order as grasp_group
    return flat_grasps, flat_scores, flat_collision
    # return grasp_list, score_list, collision_mask_list # Old return values, not used anymore


def eval_grasp(grasp_group, models, dexnet_models, poses, config, table=None, voxel_size=0.008, TOP_K = 50):
    '''
    **Input:**
    
    - grasp_group: GraspGroup instance for evaluation.

    - models: in model coordinate

    - dexnet_models: models in dexnet format 
    
    - poses: from model to camera coordinate

    - config: dexnet config.
    
    - table: in camera coordinate

    - voxel_size: float of the voxel size.

    - TOP_K: int of the number of top grasps to evaluate.
    '''

    #1. NMS grasps and associate with object
    num_models = len(models)
    ## grasp nms
    grasp_group = grasp_group.nms(0.03, 30.0/180*np.pi)

    ## assign grasps to object
    # merge and sample scene
    # A scene is a (N, 3) point cloud and an associate (N, 1) array which has the object index of each point.
    model_trans_list = list()
    seg_mask = list()
    for i,model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        # Transform model points to camera coordinates
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        # Create a mask labeling each point with its object index
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)
    # Concatenate all transformed model points into one scene point cloud

    # assign grasps
    # Compute closest points on scene (N, 3) and then look up object with (N, 1) mask
    indices = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[indices]

    # 
    """
    From paper: For each scene, we randomly pick around 10 objects from our whole object set

    From ChatGPT:
    Grasps should have an inital score which represents a preliminary confidence in the grasp quality. 
    Evaluting all possible grasps is very expensive! so better to cut down and select an managable qty
    to do expensive physics evaluation on.

    The inital score is confidence (higher is better)
    The eval score is friction coefficient (lower is better) or -1

    This is designed to take in alot of grasps then...    
    """

    """
    Why filter by object first?
        If you only filter by score globally, you might end up with most or all top grasps coming from just one or two objects
        (e.g., the easiest or most graspable ones). This would not fairly evaluate the scene, especially if you want to test
        grasping performance across all objects.
    """
    #2. Only keep top 10 grasps per object to ensure diversity in scene eval, this may result in less than TOP_K grasps thought...

    pre_grasp_list = list()
    for i in range(num_models):
        # Select grasps assigned to object i
        grasp_i = grasp_group[model_to_grasp==i]
        # Sort these grasps by their score (descending) 
        grasp_i.sort_by_score()
        # Keep only the top 10 grasps for this object (as an array)
        pre_grasp_list.append(grasp_i[:10].grasp_group_array)

    # Stack all top grasps from all objects into a single array
    # could be around 10 * 10, if dense grasp labels
    all_grasp_list = np.vstack(pre_grasp_list)


    #3. Only keep the top K grasps by thresholding with min score, across all objects

    # Sort all grasps by their score (column 0), descending, and get their indices
    remain_mask = np.argsort(all_grasp_list[:,0])[::-1]

    # Find the score of the 50th best grasp (or the lowest if fewer than 50).
    # This is used as a threshold for selecting top grasps.
    min_score = all_grasp_list[remain_mask[min(TOP_K-1,len(remain_mask) - 1)],0]


    # Further filter grasps based on the minimum score
    grasp_list = []
    for i in range(num_models):
    # For each object, create a mask for grasps with score >= min_score
        remain_mask_i = pre_grasp_list[i][:,0] >= min_score
        # Keep only those grasps for this object
        grasp_list.append(pre_grasp_list[i][remain_mask_i])

    # grasp_list = pre_grasp_list

    #4. Using scene point cloud, check for collision and empty grasps

    ## collision detection
    # If a table is provided, add its points to the scene (N + T,3)
    if table is not None:
        scene = np.concatenate([scene, table])

    # Check for collisions and empty grasps, and convert to Dex-Net format
    # grasps that are in the empty_list will be added to dexgrasp_list as None
    # which will skip below. You skip grasps that in in colission or empty
    collision_mask_list, empty_list, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_dexgrasps=True)
    
    #5. Evaluate force closure quality of the grasps

    ## evaluate grasps
    # score configurations
    force_closure_quality_config = dict()
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])
    # get grasp scores
    score_list = list()
    
    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        collision_mask = collision_mask_list[i]
        dexgrasps = dexgrasp_list[i]
        scores = list()
        num_grasps = len(dexgrasps)
        for grasp_id in range(num_grasps):
            if collision_mask[grasp_id]:
                scores.append(-1.)
                continue
            if dexgrasps[grasp_id] is None:
                scores.append(-1.)
                continue
            grasp = dexgrasps[grasp_id]
            score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)
            scores.append(score)
        score_list.append(np.array(scores))

    """
    The grasps list are the grasps that were evaluated
    Score list is the score for each grasp:
        Empty -> -1
        Collision -> -1
        Force closure Fail -> -1
        Force closure Success -> 0.2, 0.4, 0.6, 0.8, 1.0, 1.2 (lower is better)
    Collision list is if it's in collision

    I wouldn't say this is the cleanest, fastest eval way, but I guess its works!
    """

    # the length of each list is equal to the number of objects in the scene!
    return grasp_list, score_list, collision_mask_list
