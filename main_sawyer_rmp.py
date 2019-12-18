import numpy as np
# import robosuite as suite
from robosuite import Env_SawyerRmp

if __name__ == "__main__":

    # initialize the task
    env = Env_SawyerRmp(has_renderer=True,
                        ignore_done=True,
                        use_camera_obs=False,
                        control_freq=100, )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # exit()

    # do visualization
    for i in range(5000):
        action = np.random.randn(env.dof)
        action = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0]

        obs, reward, done, _ = env.step(action)

        di = env.get_obv_for_planning()

        J = di["f_jcb"][6]
        dJ = di["f_jcb_dot"][6]

        q = di["joint_pos"]
        dq = di["joint_vel"]

        env.render()
