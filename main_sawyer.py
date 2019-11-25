import numpy as np
# import robosuite as suite
from robosuite import SawyerRmp

if __name__ == "__main__":

    # initialize the task
    env = SawyerRmp(
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # do visualization
    for i in range(5000):
        action = np.random.randn(env.dof)
        action = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
        # print(action)
        obs, reward, done, _ = env.step(action)
        env.render()
