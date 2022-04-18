import os
import gym
import time
from torch import float64
import torchvision
import numpy as np
from PIL import Image
from selenium import webdriver
import matplotlib.pyplot as plt
from gym.spaces import Discrete, Box
from torchvision import transforms
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


class Env(gym.Env):
    
    VISU = True
    
    def __init__(self):
        self.driver = self.init_driver()
        self.action_space = Discrete(4)
        self.observation_space = Box(0.0, 1.0, shape=(225, 360), dtype=np.float64)
    
    def take_screenshot(self):
        path = os.path.abspath(os.getcwd()) + '/screenshot.png'
        self.driver.get_screenshot_as_file(path)
        img = Image.open(path)
        img = img.resize((360, 225))
        transform = torchvision.transforms.Compose([
            transforms.Grayscale(), 
            transforms.ToTensor()])
        os.remove(path)
        return np.squeeze(transform(img).numpy())
    
    def render(self):
        VISU = True
    
    def reset(self):
        if self.driver.current_url == 'chrome://dino/':
            self.driver.refresh()
        else:
            try:
                self.driver.get('chrome://dino/')
            except WebDriverException:
                pass
        self.driver.execute_script("document.visibilityState = 'visible';")
        ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        self.done = False
        self.driver.execute_script('console.clear()')
        self.observation = self.take_screenshot()
        return self.observation

    def close(self):
        print('Closing driver')
        self.driver.close()
    
    def init_driver(self):
        print('Starting driver...', end='', flush=True)
        
        options = webdriver.ChromeOptions()
        if Env.VISU:
            options.add_argument('--kiosk')
            options.add_argument("--incognito")
            options.add_argument('disable-infobars')
            options.add_argument('disable-extensions')
            options.add_argument('disable-notifications')
            options.add_experimental_option("useAutomationExtension", False)
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
        else: options.add_argument('headless')
        caps = DesiredCapabilities.CHROME
        caps['goog:loggingPrefs'] = { 'browser':'ALL' }
        driver = webdriver.Chrome(options = options, desired_capabilities = caps)

        print('OK')
        return driver
       
    def update_done(self):
        self.driver.execute_script('console.log(Runner.instance_.playing);')
        if 'true' in dict(self.driver.get_log('browser')[-1])['message']:
            self.done = False
        else: self.done = True
        self.driver.execute_script('console.clear()')
    
    def step(self, action: int): # add default value of the action (no keyboard clicks)
        if action == 0:
            pass
        elif action == 1:
            ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        elif action == 2:
            ActionChains(self.driver).key_down(Keys.ARROW_DOWN).perform()
        elif action == 3: 
            ActionChains(self.driver).key_up(Keys.ARROW_DOWN).perform()
        self.update_done()
        self.observation = self.take_screenshot()
        self.reward = self.compute_reward(self.observation)
        self.info = None
        return self.observation, self.reward, self.done, self.info
    
    def compute_reward(self, observation):
        pass

if __name__ == '__main__':
    env = Env()
    for i_episode in range(20):
        observation = env.reset()
        start = time.time()
        for t in range(100):
            print(t, start-time.time())
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
env.close()
