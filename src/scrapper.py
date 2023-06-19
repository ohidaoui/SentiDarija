from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

import pandas as pd


comments_by_video = {}
driver_path = "path_to_chromedriver"

# Create a new instance of the Chrome driver
driver = webdriver.Chrome(driver_path)

# Youtube videos IDs list
video_ids = []
SCROLL_PAUSE_TIME = 2

for video_id in video_ids:
    
    # Set the URL of the YouTube video to be scraped
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Navigate to the YouTube video page
    driver.get(url)

    # Wait for the page to load
    time.sleep(5)
    
    # Scroll down to the bottom of the page to load all the comments
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        
    # Find all the comment elements
    comment_elems = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text")))

    # Extract the comment text from each element and save them by id
    comments = [elem.text for elem in comment_elems]
    comments_by_video[video_id] = comments
    
# Close the browser
driver.quit()

# Concatenate all the scrapped comments
all_comments = []
for comments in comments_by_video.values():
    all_comments += comments

# Create a dataframe to store the comments
df = pd.DataFrame({'comments': all_comments, 'polarity': ''})
df.comments = df.comments.apply(lambda txt: txt.replace('\n', ''))

# Remove duplicates and save the dataframe
df.drop_duplicates().to_csv('scrapped_comments.csv', index=False)
