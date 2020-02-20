import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string

class Darias(Robot):
    """Darias is a hunky bimanual robot."""

    def __init__(self):
        super().__init__(xml_path_completion("/home/yanhua/ip2/robottasksim/robosuite/models/assets/robots/darias/darias.xml"))

        self.bottom_offset = np.array([0, 0, -0.913])
        self.left_hand = self.worldbody.find(".//body[@name='left_endeffector_link']")

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./geom[@type='box']")
        print("the node is", node)
        #node.set("pos", array_to_string(pos - self.bottom_offset))
        node.set("pos", [0.73398872, - 0.00274217,  0.87])
        print("the new node is", node)



    @property
    def dof(self):
        return 22

    @property
    def joints(self):
        out = []
        for s in ["R_", "L_"]:
            out.extend(s + a for a in ["SFE", "SAA", "HR", "EB", "WR", "WFE", "WAA", "SMS", "SMP", "SMD",
                                       "RIS", "RIP", "RID", "MIS", "MIP", "MID", "INS", "INP", "IND", "THS",
                                       "THP", "THD"])
        return out

    @property
    def init_qpos(self):
        # Arms ready to work on the table
        return np.array([
            0.535, -0.093, 0.038, 0.166, 0.643, 1.960, -1.297,
            -0.518, -0.026, -0.076, 0.175, -0.748, 1.641, -0.158,
            0.535, -0.093, 0.038, 0.166, 0.643, 1.960, -1.297, 0.535
        ])

        # Arms half extended
        return np.array([
            0.752, -0.038, -0.021, 0.161, 0.348, 2.095, -0.531,
            -0.585, -0.117, -0.037, 0.164, -0.536, 1.543, 0.204,
            0.752, -0.038, -0.021, 0.161, 0.348, 2.095, -0.531, 2.095
        ])

        # Arms fully extended
        return np.zeros(22)
