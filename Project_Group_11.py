import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt 
from streamlit_folium import folium_static
import folium
import geopandas as gp
from PIL import Image 

st.title('Project - Question 1: Profiling Customers in a Self-Service Coin Laundry Shop')

st.markdown('Prepared by:')
st.markdown('1. Nurul Syahirah Binti Mohd Arqam (1191302465)')
st.markdown('2. Koogan Letchumanan (1181102004)')
st.markdown('3. Kan Chern Hann (1181101441)')

part = st.selectbox('Select a part to view the analysis', ['Exploratory Data Analysis', 'Association Rule Mining', 'Clustering Analysis', 'Regression Models', 'Classification Models'])

if(part=='Exploratory Data Analysis'):
    st.header('1. Exploratory Data Analysis')

    st.markdown("**a) What gender is likely to go to the laundry shop at night?**")
    img1a = Image.open('1a.jpeg')
    st.image(img1a)
    st.success("Based on the figure above, the number of customers who are female is higher. Thus, females are more likely to come to the laundry shop at night compared to male.")
    
    st.markdown("**b) Is there a relationship between basket size and customers with kids?**")
    img1b = Image.open('1b.jpeg')
    st.image(img1b)
    st.success("There is no association between the basket size and customers with kids as there is no difference in the trend. This could also be due to the class imbalance where there is an overwhelming amount of big sized baskets used compared to small baskets.")

    st.markdown("**c) Is there a relationship between basket size and race?**")
    img1c = Image.open('1c.jpeg')
    st.image(img1c)
    st.success("Similar to the question above, there is no association found between basket size and race of customer as there is no difference in the trend.")

    st.markdown("**d) Which combination of Washer No. and Dryer No. are used the most by the customers?**")
    img1d = Image.open('1d.jpeg')
    st.image(img1d)
    st.success("The combination of washer and dryer used most by customers is washer number 3 and dryer number 7. The least used washer and dryer together is washer number 4 and dryer number.")

    st.markdown("**e) Does a specific age group wear a specific shirt color?**")
    img1e = Image.open('1e.jpeg')
    st.image(img1e)
    st.success("Based on the group bar chart, the trend of shirt colour and age group is similar with one another. Thus, we are not able to conclude that a specific age group wears a specific colour.")

    st.markdown("**f) Did weather information impact the sales?**")
    st.success("By finding the correlation between rainfall millimeters and number of daily sales, it is found that there is almost none to very weak negative correlation. This might be due to the use of a random rainfall dataset that is unreliable. As no further context on the laundry dataset is given, we weren't able to retrieve the correct dataset to supplement our analysis.")

elif(part=='Association Rule Mining'):
    st.header('2. Association Rule Mining')
    st.markdown("**a) What are the frequent pieces of clothing(s) worn by customers in the laundry shop?**")
    st.success("By applying the association rule mining algorithm, we are able to identify the frequent combinations of outfits that customers wore to the laundry shop. With the minimum confidence level set at 70%, and minimum support level at 25%, 6 associations are found. Customers are frequently wearing long pants and short sleeve shirts alone. Customers wearing long pants, as well as short pants, are frequently seen with short sleeve shirts. Black pants worn by customers are also often long.")

    st.markdown("**b) What are the frequent types of customers who go to the laundry shop?**")
    st.success("The frequent types of customers who go to the laundry shop are also identified. With minimum support of 5%, and minimum confidence of 40%, there are 3 associations found. Customers who are Malay are most likely to be fat with a support level of 11.5%, and confidence level of 41.8%. Customers who are in their 40s and fat are most likely to be Indian with a support level of 5.6%, and confidence level of 4.2%. While customers who are in their 40s and Indian are most likely to have a moderate body type with a support level of 5.4%, and confidence level of 43.2%.")

    st.markdown("**c) What are the frequent types of customers who go to the laundry shop at night?**")
    st.success("The frequent types of customers who go to the laundry shop at night are also identified. In this context, night is from 12pm until 6am. With minimum support of 25%, and minimum confidence of 50%, there are 3 associations found. Customers who go to the laundry shop at night are most likely to be females with a support level of 5.2%, and confidence level of 52.5%. Those customers who are fat are more likely to be male while female customers are more likely to have moderate body type.")

    st.markdown("**d) What are the frequent types of customers who choose Washer No. 3 and Dryer No. 7?**")
    st.success("As the most used washer and dryer combination is number 3 and 7 respectively, the frequent types of customers who choose this combination are identified. With minimum support of 25%, and minimum confidence of 50%, there are 3 associations found. The types of customers are similar to the types of customers who go to the laundry at night with the variation in the support, confidence and lift levels. Thus, the customers are more likely to be females, customers who are fat are more likely to be male, and customers who are female are more likely to have moderate body type.")

elif(part=='Clustering Analysis'):
    st.header('3. Clustering Analysis')
    st.markdown("**a) Customer segmentation based on race, gender, and age by decade.**")
    st.success("By using clustering analysis, we were able to do customer segmentation based on race, gender, and age by decade. As the features used in this analysis are categorical variables, K-Modes clustering algorithm is used. From the elbow method, the optimal number of clusters to have is 4. Cluster 1 has the most number of customers in the cluster, followed by Cluster 0 and Cluster 3. The cluster with the least number of customers is Cluster 2. In Cluster 0, there’s an overwhelming majority of customers who are Malay. Most of the customers in Cluster 0 are also female. Cluster 0 has the most number of customers who are in their 50s. Majority of the Chinese customers are found in Cluster 1. The cluster also has more male customers than female. Most of the customers who are in their 30s are also found in Cluster 1.")

    st.success("As for Cluster 2, no Chinese customers are found in the cluster while the majority of the customers in the cluster are Indian. The cluster also has only male customers, no female customers are grouped in the cluster. The customers who are in their 40s are more likely to be found in Cluster 2 where no customers in their 30s are found. In the last cluster, Cluster 3, most of the customers grouped in the cluster are Indian. No male customers are found in Cluster 3, only female customers. Majority of the customers who are in their 40s are in Cluster 3 while there are no customers who are in their 50s.")

    st.markdown("**b) Grouping baskets used by customers based on their sizes and colours.**")
    st.success("When grouping the baskets used by customers based on their sizes and colours, it is important to note the class imbalance in basket size. There are more big baskets used by customers compared to smaller baskets. By doing cluster analysis, we were able to take 3 as the optimal number of clusters. The number of baskets in each cluster is the highest in Cluster 0, followed by Cluster 1, and Cluster 2. In all the clusters, the majority of the baskets are big. As for the basket colour, only white baskets are found in Cluster 1 and only blue baskets are found in Cluster 2. Cluster 0 has all the colours of the basket except for white and blue.")

elif(part=='Regression Models'):
    st.header('4. Regression Models')
    st.markdown("**Predict the age range of customers.**")

elif(part=='Classification Models'):
    st.header('5. Classification Models')
    st.markdown("**a) Predict the gender of customers.**")
    st.success("First of all, feature selection is performed to find the best features to predict the gender. Here, Chi2 feature selection is used to perform the feature selection. The reason why Chi2 feature selection is used is due to the data. Chi2 feature selection is best used on categorical data input and categorical output. Since the input and output of the data is mostly categorical, therefore Chi2 feature selection is used. The bar chart below shows the 10 best results of the feature selection with the highest being Pants_Colour meaning Pants_Color contributes the highest in predicting the gender of the customer. SelectKBest is used to select the best 10 features for future model evaluation.")
    
    st.success("Next, after selecting the features, model evaluation is performed to find out which classification models are best used to predict the targeted feature. Three classification models are used for comparison, namely Decision Tree classification, Naive Bayes classification and Support Vector Machine classification. The models are separated into before hyperparameter tuning and after hyperparameter tuning. The best combination of hyperparameters for the classification models are found using the Grid Search method. The values of the hyperparameters obtained from the result are used for the tuned model. The baseline model has the default values of hyperparameters. The bar chart shows the accuracy of the models.")
    
    st.success("From the graphs above, we can see that all three classification are similar in ROC curve, with all giving positive results. The AUC scores are also quite similar with all three models having AUC scores of over 0.7. This indicates that the gender is quite dependent on the features.")
    
    st.success("From the observation, we can see that Decision Tree has the best accuracy before and after tuning, SVM shows similar accuracy after tuning. For Naive Bayes, the accuracy before tuning is higher than after tuning, this is suspected to be due to the bias of the model after tuning. We conclude that Decision Tree is the most suitable model to be used on this prediction.")
    
    st.markdown("**b) Predict the basket size used by customers.**")

    st.success("Similarly, Chi2 feature selection was used and SelectKBest method was used to select the 10 best features to predict the basket size. This time attire contributes the highest score to predicting the basket size.")

    st.success("However this time, due to the imbalanced class distribution of the basket size, SMOTE technique is used by oversampling the minority class. After this, the same procedure is conducted for this prediction as well. Model selection using the three same classification methods is used.")

    st.success("From the observation, we can see that the Decision Tree again shows the highest accuracy before and after tuning among the three models with SVM coming close after tuning, however Naive Bayes accuracy is low. Therefore, we conclude that the Decision Tree is the most suitable model for predicting the basket size.")
    
    st.success("The result is quite different from the previous prediction. This time from the ROC curve we noticed that most of the classification models were close to the center. The AUC score is also mostly in between 0.5 to 0.6. This indicates that the feature has less contribution to the outcome of the basket size meaning the basket size is an independent outcome.")