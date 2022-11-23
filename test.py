import os

from rand_param_envs.gym import error, spaces
from rand_param_envs.gym.utils import seeding
import numpy as np
from os import path
from rand_param_envs import gym
import six

try:
    from rand_param_envs import mujoco_py
    from rand_param_envs.mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.MjModel(fullpath)
        self.data = self.model.data
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        self.obs_dim = observation.size
        # print("observation")
        # print(observation)

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)
        # print("actuator_ctrlrange")
        # print(self.model.actuator_ctrlrange)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def do_simulation(self, ctrl, n_frames):
        ctrl[0] = 1.
        self.model.data.ctrl = ctrl
        for _ in range(n_frames):
            self.model.step() 
            # 这部分是引擎在做，我完全不掌握（可能部分信息在MjModel里，部分信息在导入的XML文件里）

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        # s = self.state_vector()
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
        #             (height > .7) and (abs(ang) < .2))
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def print_info(self):
        # print("observation space")
        # print(self.observation_space)
        # print("action")
        # print(self.action_space)
        print("model.data.qpos")
        print(self.model.data.qpos)
        print("model.data.qvel")
        print(self.model.data.qvel)
        # print("model.data.ctrl")
        # print(self.model.data.ctrl)
        # print("model.data.act")
        # print(self.model.data.act)
        print("self.model.data.sensordata")
        print(self.model.data.sensordata)
        # print(self.model.body_mass)
        # print('\n')
        # print(self.model.body_inertia)
        # print('\n')
        # print(self.model.dof_damping)
        # print('\n')
        # print(self.model.geom_friction)
        # print('\n')
        # print(self.model.actuator_ctrlrange)
        pass

if __name__ == "__main__":
    model_path = "/home/chenzhiyuan105/unitree_mujoco/data/a1/xml/a1.xml"
    # model_path = "/home/chenzhiyuan105/oyster/rand_param_envs/rand_param_envs/gym/envs/mujoco/assets/hopper.xml"
    env = MujocoEnv(model_path, 4)
    env.print_info()