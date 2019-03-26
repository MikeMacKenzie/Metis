## Metis Project: Building a classification model
### Tools: SQL, AWS, Python (sklearn, imblearn, statsmodels, tf-idf vectorizer, pandas, numpy, matplotlib, seaborn, regular expressions).

The broad purpose of this project was to build a classification model. I worked with yelp data that I obtained from Kaggle (https://www.kaggle.com/yelp-dataset). It is a relatively large relational database with multiple tables. The three tables I focused on were yelp_business (54,000 restaurants), yelp_review (3+ million reviews), and yelp_user (1+ million users).  In the business table, one of the columns is if the restaurant is listed as open for business or not, this was my outcome measure, that is, the key variable I would train my model to predict.

**Methods**  
The first step was creating a local PostgreSQL database to work with a subset of the data. After creating this database, I randomly selected 5000 restaurants from the yelp_business table and left joined on the yelp_review table (matching on business_id) and then left joined the yelp_user table onto that table (matching on user_id). (After working locally, I completed the same workflow on an AWS EC2 instance with the full dataset.) 

At this point I began feature engineering. One feature I created was the difference between a specific user rating for a given restaurant and that user’s average rating overall (across all restaurants). My reasoning was that a low rating from a user who typically gives high ratings may be a stronger indicator of a negative experience than a low rating from someone who typically give a low rating. 

Another feature was the review text.  I concatenated all reviews together for each restaurant, preprocessing the text by removing punctuation and making everything lowercase and created tf-idf vectors for each document. After much experimentation I found that the following hyperparameters led to the best performance: 8000 max features, basic English stop words, n-gram of 1, and a max df of 0.50. 

The last step before modelling was dealing with class imbalance. In this dataset about 15% of restuarants were listed as closed. To help deal with this imbalance I over sampled from the minority class at a ratio of 0.50.

**Results and Discussion**  
Now comes model training. I tried many different models but, in the end, a basic logistic regression model performed best. This modeled yielded an accuracy of 83% and an F1 score of .64. See the confusion matrix below. 

<p>
    <img src="https://github.com/MikeMacKenzie/Metis/blob/master/logistic_regression_yelp/Project03cm.svg" width="500" alt="cm"/>
    <br>
    <em></em>
</p>

There were several features having significant predictive power. For example, there was a negative relationship between number of reviews and being listed as closed. This is not surprising given that a restaurant that is older, all else being equal, will have more reviews than a younger restaurant. 

Another finding was related to length of text reviews.  The average text review length (in characters) for a restaurant was positively associated with closure. In other words, having longer reviews on average increased the likelihood of being listed as closed on Yelp. Perhaps people tend to ramble for longer when they have a terrible experience as opposed to a wonderful experience?

The difference between a given user rating and that user’s average rating was also a significant predictor. Specifically, if user ratings for a restaurant were lower, rather than higher, than those users’ overall average ratings (for all restaurants) then closure was more likely.

Lastly, there were a few words from the natural language processing that have some predictive power. For example, “options,” “welcoming,” and “beautiful” were negatively associated with closure where as “sad” was positively associated with being closed. 

**Conclusion**  
There could be many potential reasons why a restaurant would be listed as closed on Yelp. Perhaps the atmosphere is not beautiful, the staff is not welcoming, or there are not a lot (or too many) options. There could be other reasons, however, that might not be easily captured in the provided data. If a restaurant moves location it could be listed as “closed” at the current location even though it is not technically out of business. Maybe the owners have delicious food and are super friendly but just have financial troubles and therefore need to shut down.  
All in all, this was an enjoyable experience for me and I feel I learned a lot! 
 


