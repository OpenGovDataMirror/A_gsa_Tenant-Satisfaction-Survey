
from db import connection
import sentiment.do

# issue:  pass year paramater, and maybe just have get function with survey paramater
df = connection.get_css()

#sentiment, saves file as csv
sentiment.do.css(df,save=True)


