from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time

service = Service()
driver = webdriver.Chrome(service=service)

url = 'https://noises.online/'

driver.get(url)

time.sleep(2)  

try:
    element = driver.find_element(By.ID, 'paper')
    driver.execute_script("""
        var element = arguments[0];
        element.parentNode.removeChild(element);
    """, element)
    print("Div removed.")
except Exception as e:
    print("Div not found or error:", e)

input("Press Enter to close...")

driver.quit()
