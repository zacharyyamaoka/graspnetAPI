__author__ = 'mhgou, cxwang and hsfang'

import numpy as np
import os
import time
import pickle
import open3d as o3d
import copy

from .graspnet import GraspNet
from .grasp import GraspGroup
from .utils.config import get_config
from .utils.eval_utils import (
    get_scene_name,
    create_table_points,
    parse_posevector,
    load_dexnet_model,
    transform_points,
    compute_point_distance,
    compute_closest_points,
    voxel_sample_points,
    topk_grasps,
    get_grasp_score,
    collision_detection,
    eval_grasp,
    eval_all_grasp,
)
from .utils.xmlhandler import xmlReader
from .utils.utils import generate_scene_model, Timer

TIMER = Timer(verbose=False)

view_params = {
    "front": [0.0, 0.0, -1.0],   # direction camera is looking
    "lookat": [0.0, 0.0, 0.5],   # point camera is looking at
    "up": [0.0, -1.0, 0.0],      # camera's up direction
    "zoom": 0.5              # camera zoom - ADJUST THIS larger values zoom in, smaller values zoom out
}

class GraspNetEval(GraspNet):
    '''
    Class for evaluation on GraspNet dataset.
    
    **Input:**

    - root: string of root path for the dataset.

    - camera: string of type of the camera.

    - split: string of the date split.
    '''
    def __init__(self, root, camera, split = 'test'):
        super(GraspNetEval, self).__init__(root, camera, split)
        # Add cache for scene models and sampled points
        self._scene_cache = {
            'scene_id': None,
            'model_list': None,
            'dexmodel_list': None,
            'model_sampled_list': None
        }

    def get_scene_models(self, scene_id, ann_id):
        '''
            return models in model coordinate
        '''
        model_dir = os.path.join(self.root, 'models')
        # print('Scene {}, {}'.format(scene_id, camera))
        scene_reader = xmlReader(os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'annotations', '%04d.xml' % (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        model_list = []
        dexmodel_list = []
        for posevector in posevectors:
            obj_idx, _ = parse_posevector(posevector)
            obj_list.append(obj_idx)
        for obj_idx in obj_list:
            model = o3d.io.read_point_cloud(os.path.join(model_dir, '%03d' % obj_idx, 'nontextured.ply'))
            dex_cache_path = os.path.join(self.root, 'dex_models', '%03d.pkl' % obj_idx)
            if os.path.exists(dex_cache_path):
                with open(dex_cache_path, 'rb') as f:
                    dexmodel = pickle.load(f)
            else:
                dexmodel = load_dexnet_model(os.path.join(model_dir, '%03d' % obj_idx, 'textured'))
            points = np.array(model.points)
            model_list.append(points)
            dexmodel_list.append(dexmodel)
        return model_list, dexmodel_list, obj_list

    def get_model_poses(self, scene_id, ann_id):
        '''
        **Input:**

        - scene_id: int of the scen index.

        - ann_id: int of the annotation index.

        **Output:**

        - obj_list: list of int of object index.

        - pose_list: list of 4x4 matrices of object poses.

        - camera_pose: 4x4 matrix of the camera pose relative to the first frame.

        - align mat: 4x4 matrix of camera relative to the table.
        '''
        scene_dir = os.path.join(self.root, 'scenes')
        camera_poses_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'camera_poses.npy')
        camera_poses = np.load(camera_poses_path)
        camera_pose = camera_poses[ann_id]
        align_mat_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'cam0_wrt_table.npy')
        align_mat = np.load(align_mat_path)
        # print('Scene {}, {}'.format(scene_id, camera))
        scene_reader = xmlReader(os.path.join(scene_dir, get_scene_name(scene_id), self.camera, 'annotations', '%04d.xml'% (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        pose_list = []
        for posevector in posevectors:
            obj_idx, mat = parse_posevector(posevector)
            obj_list.append(obj_idx)
            pose_list.append(mat)
        return obj_list, pose_list, camera_pose, align_mat
        
    def eval_scene(self, scene_id, dump_folder, TOP_K = 50, return_list = False,vis = False, max_width = 0.1):
        """ 
        For backward compatibility, eval all 256 annotations as described here 
        https://graspnetapi.readthedocs.io/en/latest/example_eval.html

        The only disadvantage i see is having to loop through twice, and loading all the grasps into memory...
        - ok I can have an option to pass in paths that you then load
        """

        ann_id_list = []
        grasp_group_list = []
        for ann_id in range(256):
            grasp_path = os.path.join(dump_folder,get_scene_name(scene_id), self.camera, '%04d.npy' % (ann_id,))
            if not os.path.exists(grasp_path):
                continue
            ann_id_list.append(ann_id)
            grasp_group_list.append(grasp_path)

        print(f"Loaded: {len(ann_id_list)} Skipped: {256 - len(ann_id_list)} annotations for scene {scene_id}")

        self._eval_scene(scene_id, ann_id_list, grasp_group_list, TOP_K, return_list, vis, max_width)
    
    def eval_scene_all_grasps(self, scene_id, ann_id_list, grasp_group_list, vis = False, use_cache = True):
        """ 

        THE KEY THING TO UNDERSTAND:

        Is that you need to associate a grasp with an object for analytical checking
        For this reason, internally this will create a list of lists, were len(parent list) == num of models

        My goal is to move as fast as possible! can as little as possible to get the desired result.
        The issue right now is that it does evaluation on confidence, but I want to evaluate a grasp more deterministically!
        ideally it keeps same order of grasps...

        Idea 1. what does rgb matter do?

            https://arxiv.org/pdf/2103.02184

            - The ground truth AVHs are generated by two steps. Firstly,
            we initialize an empty AVH = 0 (V*A) x img_width x img_height that all
            the possible angles, views and places are labeled negative.
            Secondly, for each ground truth grasp pose in the GraspNet1Billion dataset, we calculate the closest angle, view and
            place and then label it as positive. For all the 100 training
            scenes in GraspNet-1Billion dataset, there are about 110
            million positive samples and 64 billion negative samples
            for Kinect and 143 million positive sample and 64 billion
            negative samples for Intel RealSense captured images.

            - Ok interesting, they don't actually evaluate it, they find the nearest one it seems

            Here is the associate code: https://github.com/GouMinghao/rgb_matters/blob/main/rgbd_graspnet/data/utils/gen_label.py#L151

        ok, I can just do both... I can verify that the grasp order is the same, and that succesful grasps are calculated as such
        
        """
        #region - Preprocessing
        assert len(ann_id_list) == len(grasp_group_list), 'ann_id_list and grasp_group_list must have the same length'
        assert len(ann_id_list) <= 256, 'Only 256 annotations per scene'

        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
        list_coe_of_friction = [0.2,0.4,0.6,0.8,1.0,1.2]

        # --- Caching logic for scene models ---
        cache = self._scene_cache
        if use_cache and cache['scene_id'] == scene_id and cache['model_list'] is not None:
            model_list = cache['model_list']
            dexmodel_list = cache['dexmodel_list']
            model_sampled_list = cache['model_sampled_list']
            print("Using cached scene models")
        else:
            print("Loading scene models... this takes a sec")
            TIMER.start("get_scene_models")
            model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0) # we don't care about objects here beacuse it will change!
            TIMER.stop("get_scene_models")

            model_sampled_list = []
            for model in model_list:
                model_sampled = voxel_sample_points(model, 0.008)
                model_sampled_list.append(model_sampled)
            # Update cache
            cache['scene_id'] = scene_id
            cache['model_list'] = model_list
            cache['dexmodel_list'] = dexmodel_list
            cache['model_sampled_list'] = model_sampled_list

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        #endregion - Preprocessing

        for ann_id, grasp_group in zip(ann_id_list, grasp_group_list):

            if isinstance(grasp_group, str):
                grasp_group = GraspGroup().from_npy(grasp_group)

            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
            cache['table_trans'] = table_trans #TODO derive this on the go..

            grasp_list, score_list, collision_mask_list = eval_all_grasp(grasp_group, model_sampled_list, dexmodel_list, pose_list, config, table=table_trans, voxel_size=0.008)

            if vis:
                t = o3d.geometry.PointCloud()
                t.points = o3d.utility.Vector3dVector(table_trans)

                model_list = generate_scene_model(self.root, 'scene_%04d' % scene_id , ann_id, return_poses=False, align=False, camera=self.camera)

                # for now it seems that 1.2 can never be returned.. all values are between 0.5 to 1...?
                gg = GraspGroup(copy.deepcopy(grasp_list))



                scores = np.array(score_list)
                score_success_mask = scores != -1
                print("all scores: ", scores)
                print("raw scores [1.2 - 0.2]: ", scores[score_success_mask])
                max_score = 1.2
                min_score = 0.2

                # this doesn't work unless you remove the -1
                # also not good because it changes depending what is in the list!
                # scores_confidence = (np.max(scores) - scores) / (np.max(scores) - np.min(scores))
                scores_confidence = -1 * (scores - max_score) / (max_score - min_score)
                print("scores_normalized [0 - 1]: ", scores_confidence[score_success_mask])
                # scores = scores / 2 + 0.5 # -1 -> 0, 0 -> 0.5, 1 -> 1
                # scores[collision_mask_list] = 0.3
                # gg.scores = scores

                color_list = []

                for i in range(len(gg)):

                    if collision_mask_list[i]:
                        color_list.append((1, 0, 0)) # RED
                    elif scores[i] == -1:
                        color_list.append((0, 0, 0)) # RED
                    else:
                        val = scores_confidence[i]
                        r = 1 - val
                        g = 1
                        b = 1 - val
                        color_list.append((r,g,b)) # GREEN to WHITE

                # Yuck! its not easy to add to thr grasp group as it internally manager the grasps as arrays... not classes
                use_collision = False
                if len(gg) < 5:
                    use_collision = True
                grasps_geometry = gg.to_open3d_geometry_list(color_list, use_defaults=False, use_collision=use_collision, show_frame=True)
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)

                origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
                new_geoms = [origin_frame]
            
                g_list_list = [
                    [pcd, *grasps_geometry, *new_geoms],
                    [pcd, *grasps_geometry, *model_list, *new_geoms],
                    [*grasps_geometry, *model_list, t, *new_geoms]
                ]

                for g_list in g_list_list:

                    o3d.visualization.draw_geometries(
                        g_list,
                        width=1280,
                        height=720,
                        lookat=view_params["lookat"],
                        up=view_params["up"],
                        front=view_params["front"],
                        zoom=view_params["zoom"]
                    )
                                 
            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)


        # this should return a list of lists, in the same order of the orginal list of list of graps (aka grasp_group)
        return grasp_list_list, score_list_list, collision_list_list

    def _eval_scene(self, scene_id, ann_id_list, grasp_group_list, TOP_K = 50, return_list = False, vis = False, max_width = 0.1, use_cache = True):
        '''
        **Input:**

        - scene_id: int of the scene index.
        
        - dump_folder: string of the folder that saves the dumped npy files.

        - TOP_K: int of the top number of grasp to evaluate

        - return_list: bool of whether to return the result list.

        - vis: bool of whether to show the result

        - max_width: float of the maximum gripper width in evaluation

        **Output:**

        - scene_accuracy: np.array of shape (256, 50, 6) of the accuracy tensor.
        '''

        #region - Preprocessing
        assert len(ann_id_list) == len(grasp_group_list), 'ann_id_list and grasp_group_list must have the same length'
        assert len(ann_id_list) <= 256, 'Only 256 annotations per scene'

        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
        list_coe_of_friction = [0.2,0.4,0.6,0.8,1.0,1.2]

        # --- Caching logic for scene models ---
        cache = self._scene_cache
        if use_cache and cache['scene_id'] == scene_id and cache['model_list'] is not None:
            model_list = cache['model_list']
            dexmodel_list = cache['dexmodel_list']
            model_sampled_list = cache['model_sampled_list']
            print("Using cached scene models")
        else:
            print("Loading scene models... this takes a sec")
            TIMER.start("get_scene_models")
            model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)
            TIMER.stop("get_scene_models")

            model_sampled_list = []
            for model in model_list:
                model_sampled = voxel_sample_points(model, 0.008)
                model_sampled_list.append(model_sampled)
            # Update cache
            cache['scene_id'] = scene_id
            cache['model_list'] = model_list
            cache['dexmodel_list'] = dexmodel_list
            cache['model_sampled_list'] = model_sampled_list

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        #endregion - Preprocessing

        # Old Way
        # for ann_id in range(256):
        #     grasp_group = GraspGroup().from_npy(os.path.join(dump_folder,get_scene_name(scene_id), self.camera, '%04d.npy' % (ann_id,)))
        for ann_id, grasp_group in zip(ann_id_list, grasp_group_list):

            if isinstance(grasp_group, str):
                grasp_group = GraspGroup().from_npy(grasp_group)

            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))


            # clip width to [0,max_width]
            # grasp_group_array = [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
            gg_array = grasp_group.grasp_group_array
            min_width_mask = (gg_array[:,1] < 0)
            max_width_mask = (gg_array[:,1] > max_width)
            gg_array[min_width_mask,1] = 0
            gg_array[max_width_mask,1] = max_width
            grasp_group.grasp_group_array = gg_array

            grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list, pose_list, config, table=table_trans, voxel_size=0.008, TOP_K = TOP_K)

            # remove empty at the object level 
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            collision_mask_list = [x for x in collision_mask_list if len(x)!=0]

            if len(grasp_list) == 0:
                grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                print('\rMean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id),np.mean(grasp_accuracy[:,:]), end='')
                continue

            # concat into scene level
            # This get rid of grouping into objects
            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(score_list), np.concatenate(collision_mask_list)
            
            if vis:
                t = o3d.geometry.PointCloud()
                t.points = o3d.utility.Vector3dVector(table_trans)
                model_list = generate_scene_model(self.root, 'scene_%04d' % scene_id , ann_id, return_poses=False, align=False, camera=self.camera)
                gg = GraspGroup(copy.deepcopy(grasp_list))
                scores = np.array(score_list)
                scores = scores / 2 + 0.5 # -1 -> 0, 0 -> 0.5, 1 -> 1
                scores[collision_mask_list] = 0.3
                print(f"Number of grasps in collision: {np.sum(collision_mask_list)} / {len(collision_mask_list)}")
                gg.scores = scores
                gg.widths = 0.1 * np.ones((len(gg)), dtype = np.float32)
                grasps_geometry = gg.to_open3d_geometry_list(show_frame=True)
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)

                origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
                new_geoms = [origin_frame]

                # o3d.visualization.draw_geometries([pcd, *grasps_geometry, *new_geoms])
                # o3d.visualization.draw_geometries([pcd, *grasps_geometry, *model_list, *new_geoms])
                # o3d.visualization.draw_geometries([*grasps_geometry, *model_list, t, *new_geoms])
            
                g_list_list = [
                    [pcd, *grasps_geometry, *new_geoms],
                    [pcd, *grasps_geometry, *model_list, *new_geoms],
                    [*grasps_geometry, *model_list, t, *new_geoms]
                ]

                for g_list in g_list_list:

                    o3d.visualization.draw_geometries(
                        g_list,
                        width=1280,
                        height=720,
                        lookat=view_params["lookat"],
                        up=view_params["up"],
                        front=view_params["front"],
                        zoom=view_params["zoom"]
                    )
                    


            # sort in scene level (right its organized in object level)
            grasp_confidence = grasp_list[:,0]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[indices]

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

            #calculate AP
            # We are going to check for the top_K confidence grasps, for each coefficient of friction 
            # the grasp accuracy for example the 20 spot, is not the grasp accuracy of the 20th grasp but of the average grasp accuracy up to 20
            # grasp_accuracy[19, 2]
            # "If I take the 20 highest-confidence grasps, what fraction of them succeed under this friction?"

            grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0,TOP_K):
                    if k+1 > len(score_list): #if not enough succesful grasps... then just take the whole list...
                        grasp_accuracy[k,fric_idx] = np.sum(((score_list<=fric) & (score_list>0)).astype(int))/(k+1)
                    else:
                        grasp_accuracy[k,fric_idx] = np.sum(((score_list[0:k+1]<=fric) & (score_list[0:k+1]>0)).astype(int))/(k+1)

            print('\rMean Accuracy for scene:%04d ann:%04d = %.3f' % (scene_id, ann_id, 100.0 * np.mean(grasp_accuracy[:,:])), end='', flush=True)
            scene_accuracy.append(grasp_accuracy)


        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list

    def parallel_eval_scenes(self, scene_ids, dump_folder, proc = 2):
        '''
        **Input:**

        - scene_ids: list of int of scene index.

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - scene_acc_list: list of the scene accuracy.
        '''
        from multiprocessing import Pool
        p = Pool(processes = proc)
        res_list = []
        for scene_id in scene_ids:
            res_list.append(p.apply_async(self.eval_scene, (scene_id, dump_folder)))
        p.close()
        p.join()
        scene_acc_list = []
        for res in res_list:
            scene_acc_list.append(res.get())
        return scene_acc_list

    def eval_seen(self, dump_folder, proc = 2):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for seen split.
        '''
        res = np.array(self.parallel_eval_scenes(scene_ids = list(range(100, 130)), dump_folder = dump_folder, proc = proc))
        ap = np.mean(res)
        print('\nEvaluation Result:\n----------\n{}, AP Seen={}'.format(self.camera, ap))
        return res, ap

    def eval_similar(self, dump_folder, proc = 2):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for similar split.
        '''
        res = np.array(self.parallel_eval_scenes(scene_ids = list(range(130, 160)), dump_folder = dump_folder, proc = proc))
        ap = np.mean(res)
        print('\nEvaluation Result:\n----------\n{}, AP={}, AP Similar={}'.format(self.camera, ap, ap))
        return res, ap

    def eval_novel(self, dump_folder, proc = 2):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for novel split.
        '''
        res = np.array(self.parallel_eval_scenes(scene_ids = list(range(160, 190)), dump_folder = dump_folder, proc = proc))
        ap = np.mean(res)
        print('\nEvaluation Result:\n----------\n{}, AP={}, AP Novel={}'.format(self.camera, ap, ap))
        return res, ap

    def eval_all(self, dump_folder, proc = 2):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for all split.
        '''
        res = np.array(self.parallel_eval_scenes(scene_ids = list(range(100, 190)), dump_folder = dump_folder, proc = proc))
        ap = [np.mean(res), np.mean(res[0:30]), np.mean(res[30:60]), np.mean(res[60:90])]
        print('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(self.camera, ap[0], ap[1], ap[2], ap[3]))
        return res, ap
