import os
import gym
import time
from torch import float64
import torchvision
import numpy as np
from PIL import Image
import stable_baselines3
from selenium import webdriver
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from torchvision import transforms
from gym.envs.registration import register 
from gym.spaces import Discrete, Box, Tuple
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import JavascriptException
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


class Env(gym.Env):
    
    VISU = True
    
    def __init__(self):
        self.driver = self.init_driver()
        self.done = True
        self.reset()
        obs = self.take_screenshot()
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0.0, high=1.0, shape=obs.shape, dtype=obs.dtype)
        try:
            self.driver.get('chrome://dino/')
        except WebDriverException:
            pass

    def take_screenshot(self):
        path = os.path.abspath(os.getcwd()) + '/screenshot.png'
        self.driver.get_screenshot_as_file(path)
        img = Image.open(path)
        transform = torchvision.transforms.Compose([
            transforms.Grayscale(), 
            transforms.ToTensor()])
        os.remove(path)
        return np.ndarray.flatten(transform(img).numpy())
    
    def render(self):
        pass
    
    def reset(self):
        if not self.done:
            return
        if self.driver.current_url == 'chrome://dino/':
            print('refreshed')
            self.driver.refresh()
        self.driver.set_window_size(360, 225)
        self.driver.execute_script("document.visibilityState = 'visible';")
        ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        self.driver.execute_script('console.clear()')
        self.observation = self.take_screenshot()
        return self.observation

    def close(self):
        print('Closing driver')
        self.driver.close()
    
    def init_driver(self):
        print('Starting driver...', end='', flush=True)
        options = webdriver.ChromeOptions()
        if not Env.VISU:
            options.add_argument('headless')
        #options.add_argument('--kiosk')
        #options.add_argument("--incognito")
        #options.add_argument('disable-infobars')
        #options.add_argument('disable-extensions')
        #options.add_argument('disable-notifications')
        #options.add_experimental_option("useAutomationExtension", False)
        #options.add_experimental_option("excludeSwitches", ["enable-automation"])
        caps = DesiredCapabilities.CHROME
        caps['goog:loggingPrefs'] = { 'browser':'ALL' }
        driver = webdriver.Chrome(options = options, desired_capabilities = caps)

        print('OK')
        return driver
    
    def step(self, action: int):
        if action == 0:
            pass
        elif action == 1:
            ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        elif action == 2:
            ActionChains(self.driver).key_down(Keys.ARROW_DOWN).perform()
        elif action == 3: 
            ActionChains(self.driver).key_up(Keys.ARROW_DOWN).perform()
        try: self.driver.execute_script('console.log(Runner.instance_.playing);')
        except JavascriptException:
            pass
        if 'true' in dict(self.driver.get_log('browser')[-1])['message']:
            self.done = False
        else: self.done = True
        self.driver.execute_script('console.clear()')
        self.observation = self.take_screenshot()
        self.driver.execute_script('console.log(Runner.instance_.currentSpeed);')
        speed = self.driver.get_log('browser')[-1]['message']
        self.reward = float(speed[17:]) - 10.0
        self.info = {}
        return self.observation, self.reward, self.done, self.info


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=8):
        env = make_vec_env(lambda: Env(), n_envs=1, vec_env_cls=DummyVecEnv)
    model = DQN('MlpPolicy', env, verbose=10)
    model.learn(total_timesteps=int(2e5))
    model.save("dinoTest1")
    del model
    model = DQN.load("dinoTest1", env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        env.render()
