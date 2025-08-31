
import numpy as np
import pytest
from graspnetAPI import GraspNetEval, Grasp, GraspGroup
import open3d as o3d
import cv2


@pytest.fixture(scope="module")
def test_params():
    """Test parameters."""
    return {
        'scene_id': 2,
        'ann_id': 1,
        'camera': 'realsense',
        'fric_coef_thresh': 0.2,
        'num_grasps': 20,
        'max_width': 0.1
    }

@pytest.fixture(scope="module")
def graspnet(test_params: dict):
    """Initialize GraspNetEval instance for testing."""
    g = GraspNetEval(root='/home/bam/graspnetAPI/graspnet', camera=test_params['camera'], split='train')
    return g


@pytest.fixture(scope="module")
def loaded_grasps(graspnet: GraspNetEval, test_params: dict):
    """Load and prepare grasps for testing."""
    # Load 6D grasps with friction coefficient threshold
    _6d_grasp = graspnet.loadGrasp(
        sceneId=test_params['scene_id'],
        annId=test_params['ann_id'],
        format='6d',
        camera=test_params['camera'],
        fric_coef_thresh=test_params['fric_coef_thresh']
    )
    
    # Sort by score to get best grasps first
    _6d_grasp.sort_by_score()
    
    # Select top grasps
    selected_grasps = _6d_grasp[:test_params['num_grasps']]
    
    return {
        'full_grasps': _6d_grasp,
        'selected_grasps': selected_grasps
    }


def test_data_completeness(graspnet):
    """Test that the GraspNet data is complete."""
    assert graspnet.checkDataCompleteness()


def test_grasp_transforms(loaded_grasps):
    """
    The frame that graspnet uses is different from TCP. I had to add transforms, this verifies they are inveritable
    """

    selected_grasps: GraspGroup = loaded_grasps['selected_grasps']

    grasp = selected_grasps[0]  # Get the first grasp for testing
    assert np.allclose(grasp.T_graspnet_tcp @ grasp.T_tcp_graspnet, np.eye(4))


@pytest.mark.slow
def test_grasp_evaluation_consistency(graspnet: GraspNetEval, loaded_grasps, test_params):
    """
    eval_scene_all_grasps function is different in that it evalautes all grasps instead of just top k, and returns the grasps
    in the same order that they were passed in. This test verifies that the orders match!
    """
    selected_grasps = loaded_grasps['selected_grasps']
    
    ann_id_list = [test_params['ann_id']]
    grasp_group_list = [selected_grasps]
    
    # Evaluate grasps
    grasp_list_list, score_list_list, collision_list_list = graspnet.eval_scene_all_grasps(
        test_params['scene_id'],
        ann_id_list,
        grasp_group_list,
        vis=False,  # Set to False for testing to avoid GUI dependencies
        use_cache=True
    )
    
    # Verify structure consistency
    assert len(grasp_group_list) == len(grasp_list_list), f"Length mismatch: {len(grasp_group_list)} vs {len(grasp_list_list)}"
    
    # Verify individual grasp group consistency
    for i, (g_group, g_list) in enumerate(zip(grasp_group_list, grasp_list_list)):
        # g_group is a GraspGroup, g_list is a numpy array
        arr1 = g_group.grasp_group_array
        arr2 = g_list
        
        assert len(arr1) == len(arr2), f"Sub-list {i} length mismatch: {len(arr1)} vs {len(arr2)}"
        
        assert np.allclose(arr1, arr2), f"Sub-list {i} arrays are not close!"
        
        print(f"Sub-list {i} passed verification: {len(arr1)} grasps match.")
    
    print("Verification complete.")


@pytest.mark.visual
def test_visualization_display(graspnet: GraspNetEval, loaded_grasps, test_params):
    """Test visualization display (marked as visual test)."""
    selected_grasps = loaded_grasps['selected_grasps']
    
    geometries = []
    geometries.append(graspnet.loadScenePointCloud(
        sceneId=test_params['scene_id'],
        annId=test_params['ann_id'],
        camera=test_params['camera']
    ))
    geometries += selected_grasps.to_open3d_geometry_list()
    
    # Note: This test requires a display and user interaction
    # It's marked with @pytest.mark.visual so it can be skipped in CI
    # Uncomment the line below to enable visualization during testing
    o3d.visualization.draw_geometries(geometries)