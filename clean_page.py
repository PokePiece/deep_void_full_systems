import argparse
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
    parser = argparse.ArgumentParser(description="Open URL and remove element before interaction.")
    parser.add_argument("url", help="The URL to open.")
    parser.add_argument("selector", help="The selector to identify the element to remove.")
    parser.add_argument("--by", default="id", choices=["id", "class", "css", "xpath"],
                        help="Selector type (default: id)")
    parser.add_argument("--driver", default="./chromedriver.exe",
                        help="Path to ChromeDriver executable")

    args = parser.parse_args()
    clean_page(args.url, args.selector, args.by, args.driver)
