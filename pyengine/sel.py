from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from threading import Thread


def wait_click(driver, iden, data, ms=20):
    button = WebDriverWait(driver, ms).until(
        EC.element_to_be_clickable((iden, data))
    )
    button.click()


def main():
    driver = webdriver.Firefox()
    driver.get("https://www.24baby.nl/babynamen/leo/")
    driver.delete_all_cookies()

    wait_click(driver, By.ID, "allowcookie")
    wait_click(driver, By.XPATH, '//*[@id="babynames-like-form"]/ul/li[2]/div')
    wait_click(driver, By.XPATH, '//*[@id="babynames-like-form"]/div/button[1]')

    driver.close()


for i in range(5):
    Thread(target=main).start()