import streamlit as st
from streamlit_tags import st_tags
import hydralit_components as hc
import datetime
import snscrape.modules.twitter as sntwitter
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (30,20)
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from model_integration import *



def scrapData():
    op = st.checkbox('Add more criteria for search')
    options_ticker  = []
    username, since, until, limit = None, None, None, 0
    if not op:
        with st.form('Scraping section'):
            col1, col2, col3, col4 = st.columns((2,0.5,2,2))
            with col1:
                options_ticker = st_tags(label='Enter Keyword:',text='Press enter to add more', maxtags = 5)
            submitted = st.form_submit_button("Scrap")
            
    else:
        list_details = ['limit', 'period']
        options_details = st.multiselect(label="Select one (or multiple) keyword", options=list_details,
                                        default='limit',
                                        help="By default, the multiselect takes #BlaBlaConf for search!")
        with st.form('Scraping section'):
            col2, col3, col4 = st.columns((1.5, 0.8,2))
            if not 'username' in options_details:
                options_ticker = st_tags(label='Enter Keyword:',text='Press enter to add more', maxtags = 5)

            if "username" in options_details:
                username = col4.text_input(label="Enter username")
            
            if "limit" in options_details:
                limit  = col2.text_input('Limit number to extract')
            
            if "period" in options_details and  'username' not in options_details:
                yesterday = datetime.date.today() - datetime.timedelta(days = 1)
                today  = datetime.date.today()
                since  = col3.date_input('Start date', yesterday)
                until = col4.date_input('End date', today)
            submitted = st.form_submit_button("Scrap")
            print(options_ticker)
    if submitted:
        with st.spinner('Wait loading data in progress...'):
            df = ScrapTwitter1(options_ticker, limit, username, since, until)
            st.write(df)
            st.success(str(df.shape[0])+' ,tweets successfully loaded!')
            df.to_csv('data.csv')


def ScrapTwitter1(options_ticker, limit, username, since, until):
    list21 = options_ticker 
    tweets_list1= []
    for n, k in enumerate (list21):
        for i, tweet in enumerate (sntwitter.TwitterSearchScraper(f'{k} ').get_items()):
            if limit !=0:
                if i > int(limit):
                    break
            print(str(i)+" | "+k+" | "+ str(tweet.date))
            tweets_list1.append([k, str(tweet.date), tweet.id, tweet.content,
                                tweet.user.username, tweet.user.followersCount,
                                tweet.retweetCount, tweet.likeCount, tweet.quoteCount, tweet.lang, tweet.cashtags])

            
    tweets_df2 = pd.DataFrame(tweets_list1, columns=['Search', 'date', 'Tweet Id', 'tweet', 'username', 'followersCount', 'replyCount',
                                                     'likeCount', 'quoteCount', 'lang', 'cashtags'])
    print("List of tweets created!")
    return tweets_df2

def upload_file():
    st.subheader("Analyse dataset")
    data_file = st.file_uploader("upload file", type=["csv"])
    
    if data_file is not None:
        file_details = {'file_name': data_file.name, "file_type": data_file.type, "file_size": data_file.size}
        st.write(file_details)
        df = pd.read_csv(data_file)
        with st.expander("Expand to see data"):
            st.dataframe(df)
        st.success(str(df.shape[0])+' ,tweets successfully loaded!')
        
        return df
    
    else:
        return None

#Functions
def navBar():
    menu_data = [
        {"id": "Scrape Data From Twitter", "icon": "fab fa-twitter", "label": "Scrape Data From Twitter"},
        {"id": "Sentiment Analysis", "icon": "far fa-chart-bar", "label": "Sentiment Analysis"},
        {"id": "Most used words", "icon": "far fa copy", "label": "Most used words"},
    ]
    over_theme = {"txc_inactive": "#FFFFFF"}
    menu_id = hc.nav_bar(menu_definition = menu_data, override_theme=over_theme, home_name='Home', first_select=0)
    return menu_id


if __name__ == '__main__':

    st.set_page_config(page_title= "Sentiment Analysis", layout="wide")
    menu_id = navBar()
    if menu_id == "Home":
        st.write('')
        st.image("arija-wt-bg.png", use_column_width=False) # image here
        st.write("")

    elif menu_id == "Scrape Data From Twitter":
        scrapData()

    elif menu_id == "Sentiment Analysis":
        uploaded_df = upload_file()
        with st.form('Analysing section'):
            analyze = st.form_submit_button("Analyze")
            if analyze:

                uploaded_df = identify_reviews_language(uploaded_df, src_field='tweet')
                uploaded_df = preprocess(uploaded_df, 'tweet', 'clean_tweet', d=None, arabizi=False)
                X = uploaded_df['clean_tweet'].to_list()
                dataloader = create_data_loader(X, 128, 8)
                test_probs = bert_predict(loaded_model, dataloader)
                pred_labels = np.argmax(test_probs, axis=1)

                label_map = {0:'positive', 1:'negative', 2:'neutral'}
                uploaded_df['Sentiment'] = [label_map[code] for code in pred_labels]

                st.subheader('Tweets after sentiment analysis')
                st.dataframe(uploaded_df)

                tweets_df3 = uploaded_df.groupby(['Search', 'Sentiment']).size().reset_index (name='Counts')        
                tweets_df3['%'] = 100 * tweets_df3['Counts'] / tweets_df3.groupby('Search')['Counts'].transform('sum')

                fig = px.bar(tweets_df3, x="Search", y='%', color='Sentiment')
                c11, c22= st.columns((3, 2))
                c22.markdown('<br><br>', unsafe_allow_html=True)
                c22.dataframe(tweets_df3)
                c11.plotly_chart(fig, use_container_width=True)

    elif menu_id == "Most used words":
        uploaded_df = upload_file()
        with st.form('Wordcloud section'):
            get_wordcloud = st.form_submit_button("Get wordcloud")
            if get_wordcloud:
                uploaded_df = identify_reviews_language(uploaded_df, src_field='tweet')
                uploaded_df = preprocess(uploaded_df, 'tweet', 'clean_tweet', d=None, arabizi=False)
                X = uploaded_df['clean_tweet'].to_list()
                stopwords = set(STOPWORDS)
                stopwords.update(["user", "hashtag", "url"])
                
                # Generate a word cloud image
                wordcloud = WordCloud(stopwords=stopwords, background_color="white", width = 800, height = 500).generate(' '.join(X))
                fig, x = plt.subplots()
                fig.set_size_inches(30, 20)
                x.imshow(wordcloud)
                plt.axis('off')
                st.pyplot(fig)