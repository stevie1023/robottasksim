from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.objects import BoxObject, CylinderObject, BallObject
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import ReachTask, UniformRandomSampler


class SawyerRmp(SawyerEnv):
    """
    This class corresponds to the stacking task for the Sawyer robot arm.
    """

    def __init__(
            self,
            gripper_type="TwoFingerGripper",
            table_full_size=(0.8, 0.8, 0.8),
            table_friction=(1., 5e-3, 1e-4),
            use_camera_obs=True,
            use_object_obs=True,
            reward_shaping=False,
            placement_initializer=None,
            gripper_visualization=False,
            use_indicator_object=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=10,
            horizon=1000,
            ignore_done=False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
    ):
        """
        Args:

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        self.use_object_obs = use_object_obs

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

        # reward configuration
        self.reward_shaping = reward_shaping

        # information of objects
        # self.object_names = [o['object_name'] for o in self.object_metadata]
        self.object_names = list(self.mujoco_obstacles.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # initialize objects of interest
        cubeA = BoxObject(
            size=[0.02, 0.02, 0.02],
            pos=[0, 0.1, 0.5],
            rgba=[1, 0, 0, 1]
        )
        cylinderA = CylinderObject(
            size=[0.035, 0.20],
            pos=[0, 0.4, 0.5],
            rgba=[0, 1, 0, 1],
        )
        self.mujoco_obstacles = OrderedDict([("cubeA", cubeA), ("cylinderA", cylinderA)])
        self.n_obstacle = len(self.mujoco_obstacles)

        ballA = BallObject(
            size=[0.1],
            pos=[0, 0.1, 0.1],
            rgba=[0, 0.1, 0, 0.1],
        )

        self.mujoco_targets = OrderedDict([("ballA", ballA)])
        self.n_target = len(self.mujoco_targets)

        # task includes arena, robot, and objects of interest
        self.model = ReachTask(
            self.mujoco_robot,
            self.mujoco_obstacles,
            self.mujoco_targets
        )

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.cubeA_body_id = self.sim.model.body_name2id("cubeA")
        self.CylinderA_body_id = self.sim.model.body_name2id("cylinderA")
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        self.cubeA_geom_id = self.sim.model.geom_name2id("cubeA")
        self.CylinderA_geom_id = self.sim.model.geom_name2id("cylinderA")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # reset positions of objects
        self.model.place_objects()

        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

    def reward(self, action):
        """
        Reward function for the task.

        The dense reward has five components.

            Reaching: in [0, 1], to encourage the arm to reach the cube
            Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            Lifting: in {0, 1}, non-zero if arm has lifted the cube
            Aligning: in [0, 0.5], encourages aligning one cube over the other
            Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The sparse reward only consists of the stacking component.
        However, the sparse reward is either 0 or 1.

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 1.0 if r_stack > 0 else 0.0

        return reward

    def staged_rewards(self):

        # return (r_reach, r_lift, r_stack)
        return (0, 0, 0)

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        # low-level object information
        if self.use_object_obs:
            # position and rotation of the first cube
            cubeA_pos = np.array(self.sim.data.body_xpos[self.cubeA_body_id])
            cubeA_quat = convert_quat(
                np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw"
            )
            di["cubeA_pos"] = cubeA_pos
            di["cubeA_quat"] = cubeA_quat

            # position and rotation of the second cube
            CylinderA_pos = np.array(self.sim.data.body_xpos[self.CylinderA_body_id])
            CylinderA_quat = convert_quat(
                np.array(self.sim.data.body_xquat[self.CylinderA_body_id]), to="xyzw"
            )
            di["cylinderA_pos"] = CylinderA_pos
            di["cylinderA_quat"] = CylinderA_quat

            # relative positions between gripper and cubes
            gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["gripper_to_cubeA"] = gripper_site_pos - cubeA_pos
            di["gripper_to_CylinderA"] = gripper_site_pos - CylinderA_pos
            di["cubeA_to_CylinderA"] = cubeA_pos - CylinderA_pos

            di["object-state"] = np.concatenate(
                [
                    cubeA_pos,
                    cubeA_quat,
                    CylinderA_pos,
                    CylinderA_quat,
                    di["gripper_to_cubeA"],
                    di["gripper_to_CylinderA"],
                    di["cubeA_to_CylinderA"],
                ]
            )

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                    self.sim.model.geom_id2name(contact.geom1) in self.finger_names
                    or self.sim.model.geom_id2name(contact.geom2) in self.finger_names
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        _, _, r_stack = self.staged_rewards()
        return r_stack > 0

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to nearest object
        if self.gripper_visualization:
            # find closest object
            square_dist = lambda x: np.sum(
                np.square(x - self.sim.data.get_site_xpos("grip_site"))
            )
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.eef_site_id] = np.inf  # make sure we don't pick the same site
            dists[self.eef_cylinder_id] = np.inf
            ob_dists = dists[
                self.object_site_ids
            ]  # filter out object sites we care about
            min_dist = np.min(ob_dists)

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba
