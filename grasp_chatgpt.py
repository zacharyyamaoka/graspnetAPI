"""
Grasp and GraspGroup classes inspired by the design reflections in bam_grasp/bam_grasp/grasp.py docstring.
- Minimal 17-param representation, extensible for extra info (color, finger dims, etc.)
- Array conversion, vectorized ops, helpers for NMS, sorting, sampling, grid/heatmap, and o3d visualization.
- Does not break original graspnet representation.
"""
import numpy as np
import copy

class Grasp:
    def __init__(self, arr=None, **kwargs):
        # 17 params: [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
        if arr is not None:
            self.from_array(arr)
        else:
            self.score = kwargs.get('score', 0.0)
            self.width = kwargs.get('width', 0.0)
            self.height = kwargs.get('height', 0.0)
            self.depth = kwargs.get('depth', 0.0)
            self.rotation = np.array(kwargs.get('rotation', np.eye(3).flatten()))
            self.translation = np.array(kwargs.get('translation', np.zeros(3)))
            self.object_id = kwargs.get('object_id', 0)
            # Optional extras
            self.color = kwargs.get('color', None)
            self.finger_height = kwargs.get('finger_height', None)
            self.finger_width = kwargs.get('finger_width', None)
            self.finger_thickness = kwargs.get('finger_thickness', None)

    def from_array(self, arr):
        self.score = arr[0]
        self.width = arr[1]
        self.height = arr[2]
        self.depth = arr[3]
        self.rotation = np.array(arr[4:13]).reshape(3,3)
        self.translation = np.array(arr[13:16])
        self.object_id = int(arr[16])

    def to_array(self, full=False):
        arr = np.concatenate([
            [self.score, self.width, self.height, self.depth],
            self.rotation.flatten(),
            self.translation,
            [self.object_id]
        ])
        # Optionally add extras
        if full:
            extras = [self.color, self.finger_height, self.finger_width, self.finger_thickness]
            arr = np.concatenate([arr, np.array([e if e is not None else 0 for e in extras])])
        return arr

    def to_o3d(self):
        """
        Returns an Open3D geometry representing the grasp.
        This is a placeholder; you should implement your own visualization logic.
        """
        try:
            import open3d as o3d
            mesh = o3d.geometry.TriangleMesh.create_box(width=self.width, height=self.height, depth=self.depth)
            mesh.paint_uniform_color([1, 0, 0])
            mesh.rotate(self.rotation, center=(0, 0, 0))
            mesh.translate(self.translation)
            return mesh
        except ImportError:
            return None

    def __repr__(self):
        return f"Grasp(score={self.score:.3f}, width={self.width:.3f}, pos={self.translation}, obj={self.object_id})"

class GraspGroup:
    def __init__(self, arr=None, grasps=None):
        if arr is not None:
            self.grasps = [Grasp(arr=a) for a in arr]
        elif grasps is not None:
            self.grasps = grasps
        else:
            self.grasps = []

    def __len__(self):
        return len(self.grasps)

    def __getitem__(self, idx):
        return self.grasps[idx]

    def __repr__(self):
        return f"GraspGroup(n={len(self)})"

    @classmethod
    def from_npy(cls, path):
        arr = np.load(path)
        return cls(arr=arr)

    def save_npy(self, path):
        arr = np.stack([g.to_array() for g in self.grasps])
        np.save(path, arr)

    def add(self, grasp):
        self.grasps.append(grasp)

    def remove(self, idx):
        del self.grasps[idx]

    def to_o3d(self):
        # Recursively build o3d geometries for all grasps
        return [g.to_o3d() for g in self.grasps]

    def sort_by_score(self, reverse=True):
        self.grasps.sort(key=lambda g: g.score, reverse=reverse)

    def sample(self, n_samples):
        idxs = np.random.choice(len(self.grasps), n_samples, replace=False)
        return GraspGroup(grasps=[self.grasps[i] for i in idxs])

    def nms(self, thresh=0.05):
        """
        Non-maximum suppression: removes grasps that are too close in translation and have lower score.
        This is a simple greedy implementation.
        """
        if not self.grasps:
            return
        keep = []
        arr = np.stack([g.to_array() for g in self.grasps])
        idxs = np.argsort(-arr[:, 0])  # sort by score descending
        taken = np.zeros(len(self.grasps), dtype=bool)
        for i in idxs:
            if taken[i]:
                continue
            keep.append(self.grasps[i])
            for j in idxs:
                if i == j or taken[j]:
                    continue
                dist = np.linalg.norm(self.grasps[i].translation - self.grasps[j].translation)
                if dist < thresh:
                    taken[j] = True
        self.grasps = keep

    def to_graspnet(self):
        """
        Placeholder for conversion to graspnet frame. Implement as needed.
        """
        # Example: return a new GraspGroup with transformed grasps
        return self

    def discretize(self, grid_params):
        """
        Discretize grasps into a grid/heatmap.
        grid_params: list of (n_bucket, min, max) for each dimension (e.g. x, y, z, rx, ry, rz, width)
        Returns: heatmap (ndarray), mapping from grid to best score.
        """
        n_dims = len(grid_params)
        shape = [p[0] for p in grid_params]
        heatmap = np.zeros(shape)
        for g in self.grasps:
            arr = g.to_array()
            idxs = []
            for i, (n, minv, maxv) in enumerate(grid_params):
                v = arr[i]
                b = int(np.clip((v - minv) / (maxv - minv) * n, 0, n - 1))
                idxs.append(b)
            idxs = tuple(idxs)
            heatmap[idxs] = max(heatmap[idxs], g.score)
        return heatmap

    def transform(self, T):
        """
        Apply transformation matrix T (4x4) to all grasps.
        """
        for g in self.grasps:
            g.translation = (T[:3,:3] @ g.translation) + T[:3,3]
            g.rotation = T[:3,:3] @ g.rotation

    def to_dict(self):
        """
        Returns a list of arrays for all grasps (for serialization or export).
        """
        return [g.to_array() for g in self.grasps]
