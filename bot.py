import gym
import time
from selenium import webdriver
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
        self.driver = self.initDriver()
    
    def reset(self):
        if self.driver.current_url == 'chrome://dino/':
            self.driver.reload()
        else:
            try:
                self.driver.get('chrome://dino/')
            except WebDriverException:
                pass
        self.driver.execute_script("document.visibilityState = 'visible';")
        ActionChains(env.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
    
    def initDriver(self):
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
       
    def done(self):
        self.driver.execute_script('console.log(Runner.instance_.playing);')
        try: 
            if 'true' in dict(self.driver.get_log('browser')[-1])['message']:
                return False
            else: return True
        finally: self.driver.execute_script('console.clear()')


if __name__ == '__main__':
    env = Env()
    env.reset()
    input()