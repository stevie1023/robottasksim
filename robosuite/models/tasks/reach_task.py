from robosuite.models.tasks import Task, UniformRandomSampler
from robosuite.utils.mjcf_utils import new_joint, array_to_string


class ReachTask(Task):
    """

    """

    def __init__(self, mujoco_robot, mujoco_obstacles, mujoco_targets, initializer=None):
        """
        Args:

            mujoco_robot: MJCF model of robot model

        """
        super().__init__()

        self.merge_robot(mujoco_robot)
        self.merge_obstacles(mujoco_obstacles)
        # self.merge_target(mujoco_targets)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_obstacles(self, mujoco_obstacles):
        """Adds physical objects to the MJCF model."""

        self.mujoco_obstacles = mujoco_obstacles
        self.obstacles = []  # xml manifestation
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_obstacles.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free"))
            self.obstacles.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def merge_target(self, mujoco_targets):
        """Adds physical objects to the MJCF model."""

        self.mujoco_targets = mujoco_targets
        self.targets = []  # xml manifestation
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_targets.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free"))
            self.targets.append(obj)
            self.worldbody.append(obj)
