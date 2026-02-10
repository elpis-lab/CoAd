import genesis as gs #type: ignore

# Genesis initialization
def init_scene(use_GPU, perf_mode, viewer):
    import genesis as gs #type: ignore
    if use_GPU:
        gs.init(backend=gs.gpu, performance_mode=perf_mode)
    else:
        gs.init(backend=gs.cpu, performance_mode=perf_mode)
    
    scene = gs.Scene(show_viewer=viewer, show_FPS=False)
    return scene

# Add robot with URDF
def add_robot(scene, robotURDF, robot_pos, robot_quat):
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=robotURDF, 
            pos=robot_pos, 
            quat=robot_quat,
            fixed=True
        )
    )
    plane = scene.add_entity(gs.morphs.Plane())
    return robot

# Add primitives
def add_box(scene, box_pos, box_quat, box_size, fixed=True):
    box = scene.add_entity(
        gs.morphs.Box(
            pos=box_pos,
            quat=box_quat,
            size=box_size,
            fixed=fixed
        )
    )
    return box

def add_cylinder(scene, box_pos, box_quat, box_size, fixed=True):
    cylinder = scene.add_entity(
        gs.morphs.Box(
            pos=box_pos,
            quat=box_quat,
            size=box_size,
            fixed=fixed
        )
    )
    return cylinder

# Build genesis scene
def build_scene(scene, n_envs, env_spacing=(1.0, 1.0)):
    scene.build(n_envs=n_envs, env_spacing=env_spacing)

def visualize(scene, robot, path):
    robot.set_dofs_position(path[0])
    scene.step()
    input("Begin visualization?")
        
    for waypoint in path:
        robot.set_dofs_position(waypoint)
        scene.step()

        if len(robot.detect_collision())>0:
            print(robot.detect_collision())
