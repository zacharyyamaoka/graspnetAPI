__author__ = 'mhgou'
__version__ = '1.0'

from graspnetAPI import GraspNet, GraspNetEval
import open3d as o3d
import cv2

"""
Now I am using a python file in case there was an issue with the jupyter notebook
"""
####################################################################
graspnet_root = '/home/bam/graspnetAPI/graspnet' # ROOT PATH FOR GRASPNET
####################################################################
sceneId = 99
annId = 240
camera = 'realsense'

# initialize a GraspNet instance  
g = GraspNetEval(graspnet_root, camera=camera, split='train')

# load grasps of scene 1 with annotation id = 3, camera = kinect and fric_coef_thresh = 0.2
_6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = camera, fric_coef_thresh = 1.0)
print('6d grasp:\n{}'.format(_6d_grasp))

# visualize the grasps using open3d
geometries = []
geometries.append(g.loadScenePointCloud(sceneId = sceneId, annId = annId, camera = camera))
selected_grasps = _6d_grasp.random_sample(numGrasp = 20)
geometries += selected_grasps.to_open3d_geometry_list()
o3d.visualization.draw_geometries(geometries)

ann_id_list = [annId]
grasp_group_list = [selected_grasps]
grasp_list_list, score_list_list, collision_list_list = g.eval_scene_all_grasps(sceneId, ann_id_list, grasp_group_list, vis=True, use_cache = True)

# g._eval_scene(sceneId, ann_id_list, grasp_group_list, vis=True, max_width = 0.1, use_cache = False)

# # load rectangle grasps of scene 1 with annotation id = 3, camera = realsense and fric_coef_thresh = 0.2
# rect_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = 'rect', camera = camera, fric_coef_thresh = 0.2)
# print('rectangle grasp:\n{}'.format(rect_grasp))

# # visualize the rectanglegrasps using opencv
# bgr = g.loadBGR(sceneId = sceneId, annId = annId, camera = camera)
# img = rect_grasp.to_opencv_image(bgr, numGrasp = 20)
# cv2.imshow('rectangle grasps', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()