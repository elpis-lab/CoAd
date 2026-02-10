import numpy as np
import genesis as gs # type: ignore

class SweptVolumeCube:
    '''
    Creates a swept volume of a cube object over its possible occupancy region
    No yaw variation = enlargened rectangle
    yaw variation = enlargened rectange with rounded corners
    '''
    def __init__(self, scene, object_dims, n_envs=1, collision=True):
        self.scene = scene
        self.object_dims = object_dims
        self.collision = collision
        self.vols = None
        self.n_envs = n_envs
    
    def _compute_parameters(self, object_configs):
        object_configs = np.asarray(object_configs, dtype=np.float64)
        xdim, ydim, zdim = self.object_dims
        #x_lower, x_upper = object_configs[0]
        #y_lower, y_upper = object_configs[1]
        #z = object_configs[2][0] # No Z variation in any cases

        if object_configs.ndim == 2:
            object_configs = object_configs[None, :, :]  

        x_lower = object_configs[:, 0, 0]
        x_upper = object_configs[:, 0, 1]
        y_lower = object_configs[:, 1, 0]
        y_upper = object_configs[:, 1, 1]
        z = object_configs[:, 2, 0] 

        R = float(np.round(0.5 * np.sqrt(xdim**2 + ydim**2), 5))
        cx = 0.5 * (x_upper + x_lower)
        cy = 0.5 * (y_upper + y_lower)

        '''
        b1_size = [ (x_upper - x_lower) + 2*R, (y_upper - y_lower), zdim ]
        b2_size = [ (x_upper - x_lower), (y_upper - y_lower) + 2*R, zdim ]

        b_pos = [cx, cy, z]
        c_z = z

        corners = [
            [x_lower, y_lower, c_z],
            [x_lower, y_upper, c_z],
            [x_upper, y_lower, c_z],
            [x_upper, y_upper, c_z],
        ]
        '''
        b1_size = np.array([ (x_upper - x_lower) + 2*R,
                            (y_upper - y_lower),
                            np.full_like(cx, zdim) ]).T   # (B,3)

        b2_size = np.array([ (x_upper - x_lower),
                            (y_upper - y_lower) + 2*R,
                            np.full_like(cx, zdim) ]).T   # (B,3)

        b_pos = np.stack([cx, cy, z], axis=1)  # (B,3)

        corners = np.stack([
            np.stack([x_lower, y_lower, z], axis=1),
            np.stack([x_lower, y_upper, z], axis=1),
            np.stack([x_upper, y_lower, z], axis=1),
            np.stack([x_upper, y_upper, z], axis=1),
        ], axis=1)  # (B,4,3)
        
        return R, b_pos, b1_size, b2_size, corners
    
    def create(self, object_configs, env_idx=None):
        R, b_pos, b1_size, b2_size, corners = self._compute_parameters(object_configs)    

        b_pos0 = b_pos[0].tolist()
        b1_size0 = b1_size[0].tolist()
        b2_size0 = b2_size[0].tolist()
        corners0 = corners[0]  

        b1 = self.scene.add_entity(gs.morphs.Box(
            pos=b_pos0,
            size=b1_size0,
            quat=[1, 0, 0, 0],
            fixed=True,
            collision=True
        ))

        b2 = self.scene.add_entity(gs.morphs.Box(
            pos=b_pos0,
            size=b2_size0,
            quat=[1, 0, 0, 0],
            fixed=True,
            collision=True
        ))

        cs = []
        for p in corners0:
            cs.append(self.scene.add_entity(gs.morphs.Cylinder(
                pos=p.tolist(),
                height=self.object_dims[2],
                radius=R,
                quat=[1, 0, 0, 0],
                fixed=True,
                collision=True
            )))
        
        self.vols = [b1, b2, cs[0], cs[1], cs[2], cs[3]]
        #print("Build attempt")
        self.scene.build(n_envs=self.n_envs, env_spacing=(1.0, 1.0))
        #print("done building")
        self.scene.step()
        return self.vols
    
    def update(self, object_configs, env_idx=None):
        object_configs = tuple(tuple(v for v in row) for row in object_configs)
        if self.vols is None:
            return self.create(object_configs, env_idx=env_idx)
        
        R, b_pos, _, _, corners = self._compute_parameters(object_configs)
        b1, b2, c1, c2, c3, c4 = self.vols

        if env_idx is not None:
            b1.set_pos(pos=b_pos, envs_idx=env_idx)
            b2.set_pos(pos=b_pos, envs_idx=env_idx)
            c1.set_pos(pos=corners[:, 0, :], envs_idx=env_idx)
            c2.set_pos(pos=corners[:, 1, :], envs_idx=env_idx)
            c3.set_pos(pos=corners[:, 2, :], envs_idx=env_idx)
            c4.set_pos(pos=corners[:, 3, :], envs_idx=env_idx)
        else:
            b1.set_pos(pos=b_pos, envs_idx=None)
            b2.set_pos(pos=b_pos, envs_idx=None)
            c1.set_pos(pos=corners[0], envs_idx=None)
            c2.set_pos(pos=corners[1], envs_idx=None)
            c3.set_pos(pos=corners[2], envs_idx=None)
            c4.set_pos(pos=corners[3], envs_idx=None)

        return self.vols