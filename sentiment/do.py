# -*- coding: utf-8 -*-
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd


def css(df,save=True):
    #drop if one word response
    df= df[df.value.str.contains(' ')]
    
    sid = SentimentIntensityAnalyzer()
    
    df['COMPOUND_SENT'] = df['value'].apply(lambda x: sid.polarity_scores(x)['compound'] if pd.isnull(x)==False else None)
    if save:
        df.to_csv("css_sentiment_score.csv")
    else:
        return df