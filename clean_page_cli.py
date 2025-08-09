import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

def clean_page(url, selector, by='id', driver_path='./chromedriver.exe'):
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service)

    driver.get(url)
    time.sleep(2)

    try:
        if by == 'id':
            element = driver.find_element(By.ID, selector)
        elif by == 'class':
            element = driver.find_element(By.CLASS_NAME, selector)
        elif by == 'css':
            element = driver.find_element(By.CSS_SELECTOR, selector)
        elif by == 'xpath':
            element = driver.find_element(By.XPATH, selector)
        else:
            raise ValueError("Unsupported selector type")

        driver.execute_script("arguments[0].remove();", element)
        print("Element removed.")
    except Exception as e:
        print("Error locating or removing element:", e)

    input("Press Enter to close the browser...")
    driver.quit()

if __name__ == "__main__":
    url = input("Enter the URL to open: ")
    selector = input("Enter the element selector (ID/class/xpath/css): ")
    by = input("Selector type [id/class/css/xpath] (default: id): ") or "id"

    clean_page(url, selector, by)
