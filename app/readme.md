A responsive `Streamlit` application that incorporates three functionalities: scraping data from Twitter, performing sentiment analysis on the scraped data, and generating a word cloud.

Here’s a breakdown of its functionalities:
  
### 1. Scraping Data from Twitter:
- The app allows users to search for tweets based on a set of keywords. Users can enter keywords or hashtags to search for relevant tweets.
- The app also allows adding a limit on the number of scraped tweets and specifying the time period of the tweets to retrieve.
- It utilizes the snscrape library to retrieve tweets from Twitter.
- The scraped tweets are stored in a Pandas DataFrame and displayed in the app.
  
### 2. Uploading and Analyzing Dataset:
- Users can upload a CSV file containing tweet data for analysis.
- The uploaded file is read using Pandas, and the DataFrame is displayed in the app.
- Users can then choose to analyze the uploaded dataset by clicking the "Analyze" button.
- The app performs sentiment analysis on the tweets using our pre-trained model and displays the results.
- The analyzed tweets are displayed in a DataFrame, including an additional column indicating the sentiment (positive, negative, or neutral) of each tweet.
  
### 3. Word Cloud Generation:
- Users can also generate a word cloud.
- The app utilizes the WordCloud library to create the word cloud.
- The resulting word cloud image is displayed in the app.
