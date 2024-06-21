#!/usr/bin/env python
# coding: utf-8

# ## Abstract
# 
# 
# Obesity is a disease that affects the health of men and women, and in recent decades it had an increasing trend, 
# the WHO estimates that by 2030 more than 40% of the world’s population will be overweight and more than a 
# fifth will be obese Consequently, researchers have made great efforts to identify early the factors that influence 
# the generation of obesity. There are tools limited to the calculation of BMI, omitting other relevant factors such 
# as: if the individual has a family history of obesity, time spent on exercise routines, genetic expression profiles 
# and other factor . 
#                   While BMI is a simple measure that is very useful for populations, it can only predict risk in individuals. Afterall, BMI is a measure of size not health and so has some limits as a diagnostic tool. For example, athletes are commonly misclassified due to their high muscle mass.  In short, BMI is most useful at a population level and for determining risk (not diagnosis) at a individual level.  
# 
# Other methods of classifying obesity include measurement of waist circumference, waist to hip ratio and the Edmonton Obesity Staging System (EOSS).Waist circumference (WC) is a cheap and easy method of measurement. Waist circumference is considered a reasonable indicator of intra-abdominal or visceral fat. This fat is closely associated with increased risk of comorbidity. The National Institute for Health and Care Excellence cut off points suggest males with WC >94cm or Females with WC => 85cm are considered to be at increased risk. The World Health Organisation have identified levels of risk combining both BMI and WC.
# On the other hand, waist to hip involves two measurements and but is also cheap and easy to use on large populations. The ratio highlights if excess weight is stored around the waist resulting in increased risk of comorbidities. Males with a waist to height ratio >1.0 and Females with a weight to height ratio >0.85 are considered to be at increased risk.     
# Increasingly, the EOSS is being used in clinical settings. The EOSS diagnoses and considers the severity of obesity based on a clinical assessment of weight-related health issues, mental health and quality of life.  This is useful at an individual level and for decision-making for treatment, but it is not practical at the population level.
# Finally, biometric impedance, Magnetic Resonance Imaging, Computed Tomography and Dual Energy X-ray Absorptiometry scans are also available, but most are normally too expensive to be used on large populations.
# 
# 

# # Obesity analysis :
# ### Exploratory Data Analysis Project: Obesity Levels Based on Eating Habits and Physical Condition

#  This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77% of the data was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through a web platform.
# 
# 

#  ### Given these attributes, I approached this project with the goal of trying to find the answers to the following questions:
# 
# As the makeup of respondents is important to the conclusions derived from this dataset, what kinds of characteristics do the people in this dataset have?
# Can BMI be used as a quantitative substitute for the qualitative weight classification category?
# Which eating habit and physical condition variables are most related to obesity levels? This question has many subquestions related to individual variables and groups of variables.
# 

# # objectives 
# 1. To study the factors that contribute to obesity using statistical analysis
# 
# 2. to classify the different classes of obesity using machine learning models 

# ### More about the data
# Number of attributes : 17
# 
# Number of rows : 2111
# 
# ### Independent Variables
# 
#  Gender - (Male/Female)
#  
#  Age - In years
#  
#  Height - In meters 
#  
#  Weight - In Kgs
#  
#  family_history_with_overweight - Family history in obesity - Yes or No
#  
#  FAVC - Frequent consumption of high caloric food - Yes/No
#  
#  FCVC - Frequency of consumption of vegetables - 1 = never, 2 = sometimes, 3 = always
#  
#  NCP - Number of main meals - 1, 2, 3 or 4 meals
#  
# 
# CAEC - Consumption of food between meals - No, Sometimes, Frequently, Always 
# 
# Smoke - Does the person smoke - Yes/No
# 
# CH20 - Consumption of water daily - 1 = less than a liter, 2 = 1–2 liters, 3 = more than 2 liters 
# 
# SCC - Calories consumption monitoring - Yes/No 
# 
# FAF - Physical activity frequency - 0 = none, 1 = 1 to 2 days, 2= 2 to 4 days, 3 = 4 to 5 days 
# 
# TUE - Time using technology devices - 0 = 0–2 hours, 1 = 3–5 hours, 2 = more than 5 hours 
# 
# CALC - Consumption of alcohol - No, Sometimes, Frequently and Always 
# 
# MTRANS - Transportation used - Public Transportation, Motorbike, Bike, Automobile and Walking 
# 
# 
# ### Dependent Variables
# 
#  NObeyesdad -  Obesity level - Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III 
# 
# 
# 

# In[1]:


# pandas

import numpy as np
import pandas as pd 
import seaborn as sns


# In[2]:


df = pd.read_csv("obesity clean data kaggle 1.csv") 


# In[3]:


df.head()


# In[4]:


df


# In[5]:


# Create a new DataFrame df1 as a copy of df
df1 = df.copy()

# Perform the modifications on df1
df1['NObeyesdad'] = df1['NObeyesdad'].apply(lambda x: x.replace('_', ' '))
df1['MTRANS'] = df1['MTRANS'].apply(lambda x: x.replace('_', ' '))
df1['Height'] = df1['Height'] * 100
df1['Height'] = df1['Height'].round(1)
df1['Weight'] = df1['Weight'].round(1)
df1['Age'] = df1['Age'].round(1)


print(df1.head())


# In[6]:


df1


# In[7]:


import pandas as pd

# Assuming 'df1' is your DataFrame
exclude_columns = [ 'Age', 'Height' ,'Weight' ]

for column in df1.columns:
    if column not in exclude_columns:
        unique_values = df1[column].unique()
        print(f"Unique values in column '{column}':")
        for value in unique_values:
            print(value)
        print()  # Print an empty line after each column's unique values


# In[8]:


# Check how many duplicate rows there are
dup_df = df1[df1.duplicated()]
print(dup_df.shape)

# Drop duplicates
df1 = df1.drop_duplicates(keep='last')
print(df1.shape)


# In[9]:


df1.info()


# ### null value checking 

# In[10]:


#Check if there are any missing values
import seaborn as sn
sn.heatmap(df1.isnull(), cbar=False, yticklabels=False, cmap='icefire')


# # EDA

# In[11]:


df1


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt


# ## Graph 1 to 9

# In[13]:


# count plot on single categorical variable
sns.countplot(x ='Gender', data = df1,palette="Set2" )



# There are almost an equal number of females and males in the dataset. Data is available for slightly more men than women but this does not make it imbalanced.
# 
# 

# ## dealing with pre-label data 
# 

# In[14]:


df1.head()


# In[15]:


import numpy as np

for x in ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
    value = np.array(df1[x])
    print(x, ':', 'min:', np.min(value), 'max:', np.max(value))


# ### This is an interpretation of pre-label data 
# 
#  FCVC - Frequency of consumption of vegetables - 1 = never, 2 = sometimes, 3 = always 
# 
#  NCP  Number of main meals - 1, 2, 3 or 4 meals 
# 
#  CH20 - Consumption of water daily - 1 = less than a liter, 2 = 1–2 liters, 3 = more than 2 liters 
# 
#  FAF - Physical activity frequency - 0 = none, 1 = 1 to 2 days, 2= 2 to 4 days, 3 = 4 to 5 days 
# 
#  TUE - Time using technology devices - 0 = 0–2 hours, 1 = 3–5 hours, 2 = more than 5 hours 
# 

# In[16]:


ax=sns.countplot(x ='FCVC', data = df1,palette="Set2"  )
ax.set_title("Frequency of consumption of vegetables ")
ax.set_xlabel('fcvc')
ax.set_ylabel('count')

plt.show()


#  FCVC - Frequency of consumption of vegetables - 1 = never, 2 = sometimes, 3 = always 
#  
#  from graph we can say that most people eat veetables on regular basis 
# 

# In[17]:


ax=sns.countplot(x ='NCP', data = df1,palette="Set2"  )
ax.set_title("Number of main meals")
ax.set_xlabel('ncp')
ax.set_ylabel('count')

plt.show()



#  NCP - Number of main meals - 1, 2, 3 or 4 meals 
# 
# most people have 3 meals per day 

# In[18]:


ax=sns.countplot(x ='CH2O', data = df1,palette="Set2"  )
ax.set_title("Consumption of water daily")
ax.set_xlabel('water intake in litters')
ax.set_ylabel('count')

plt.show()



#  CH20 - Consumption of water daily - 1 = less than a liter, 2 = 1–2 liters, 3 = more than 2 liters 
# 
# 
# Drinking water habits is given into categorised in three groups: "Less than a litter", "Between 1 and 2 L", and "More than 2,
# avg is between 1 and 2 liter.
# 

# In[19]:


ax=sns.countplot(x ='FAF', data = df1,palette="Set2"  )
ax.set_title("Physical activity frequency")
ax.set_xlabel('faf')
ax.set_ylabel('count')

plt.show()

 


#  FAF - Physical activity frequency - 0 = none, 1 = 1 to 2 days, 2= 2 to 4 days, 3 = 4 to 5 days 
# 
# Respondents were asked to share their physical activity. They had to choose 1 out of 4 optional answers: "I do not have", "1 or 2 days", "2 or 4 days", and "4 or 5 days". Most people exercise 1-2 days a week

# In[20]:


ax=sns.countplot(x ='TUE', data = df1,palette="Set2"  )
ax.set_title("Time using technology devices")
ax.set_xlabel('tuf')
ax.set_ylabel('count')

plt.show()

 


# TUE - Time using technology devices - 0 = 0–2 hours, 1 = 3–5 hours, 2 = more than 5 hours 
# 
# Similarly, people were asked to state how much time they spend on using technological devices such as cell phone, videogames, television, computer, etc. They could say "0-2 hours", "3-5 hours", and "More than 5 hours". avg use of technology device is around 0-2 hours per day 
#  

# In[21]:


ax=sns.countplot(x ='CAEC', data = df1,palette="Set2"  )
ax.set_title("Consumption of food between meals")
ax.set_xlabel('caec')
ax.set_ylabel('count')

plt.show()

 


# CAEC - Consumption of food between meals - No, Sometimes, Frequently, Always 
# 
# People had to say if and how offen they eat between meals. They could answer eigher "No" (if they do not get bites between regular time for eating), or "Sometimes", "Frequently", or "Always". The data suggests that most people "sometimes" get small snacks between meals

# In[22]:


ax=sns.countplot(x ='CALC', data = df1,palette="Set2"  )
ax.set_title("Consumption of alcohol")
ax.set_xlabel('calc')
ax.set_ylabel('count')

plt.show()

 


# CALC - Consumption of alcohol - No, Sometimes, Frequently and Always  
# 
# Most people drink alcohol "sometimes", but almost a third claim they do not consume any alcoholic beverages.
# 

# In[23]:


ax=sns.countplot(x ='MTRANS', data = df1,palette="Set2"  )
ax.set_title("Transportation used")
ax.set_xlabel('mtrans')
ax.set_ylabel('count')

plt.show()

 


# MTRANS - Transportation used - Public Transportation, Motorbike, Bike, Automobile and Walking 
# 
# Most people (around 3/4) rely on public transportation. Much fewer respondents use their cars. The remainder either commute or use a bike or motorbike.
# 
# 

# 
# ### Graphs 10-13: How are respondents responding to yes/no questions?
# 
# 

# In[24]:


plt.figure(figsize=(14,10))

#Subplot regarding family history
plt.subplot(2, 2, 1)
plt.title("Number of Respondents with Family History of Overweightness")
counts = df1["family_history_with_overweight"].value_counts()
plt.bar(counts.index, counts.values, color = ['#215A73', '#dea58c'])
plt.xlabel("Family History of Overweightness?")
plt.ylabel("Number of Respondents")

#Subplot regarding consumption of high caloric food
plt.subplot(2, 2, 2)
plt.title("Number of Respondents that Frequently Consume High Caloric Food (FAVC)")
counts = df1["FAVC"].value_counts()
plt.bar(counts.index, counts.values, color = ['#215A73', '#dea58c'])
plt.xlabel("High-Calorie Food Consumption ?")
plt.ylabel("Number of Respondents")

#Subplot regarding calorie monitoring
plt.subplot(2, 2, 3)
plt.title("Number of Respondents that Monitor Calorie Consumption (SCC)")
counts = df1["SCC"].value_counts()
plt.bar(counts.index, counts.values, color = ['#dea58c','#215A73'])
plt.xlabel("Calorie Consumption Monitoring?")
plt.ylabel("Number of Respondents")

#Subplot regarding smoking
plt.subplot(2, 2, 4)
plt.title("Number of Respondents that Smoke")
counts = df1["SMOKE"].value_counts()
plt.bar(counts.index, counts.values, color = ['#dea58c','#215A73'])
plt.xlabel("Smokes?")
plt.ylabel("Number of Respondents")
plt.show()


#  #observations :
# 1. People were asked if family members suffered from overweight. Most of them replied affirmative(+1600 say yes) .
# 2. Survey respondents had to say if they eat high caloric food frequenty. There were only two possible answers: "yes" or "no".      Most of them (88%) admitted they consume high caloric food.
# 3. It seems people do not worry about the calories they get daily. On the other hand, they might not have been aware of the        nutritional value and ingredients of each food if these were not listed on the packing.
# 4. Most respondents do not smoke.(2067)
# 
# 
# 

# # Data Analysis :  understanding the nature of target variable 
# 
# ### BMI calculation

# In[25]:


# Calculate BMI using the given formula
df1['BMI'] = (df1['Weight'] / df1['Height'] / df1['Height']) * 10000

# Reorder columns to put BMI immediately after Height and Weight
df1 = df1[['Gender', 'Age', 'Height', 'Weight', 'BMI', 'family_history_with_overweight',
         'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad']]

# Check if the new column was calculated and placed correctly
df1.head()


# 
# ### Graph 14 : How are the respondents broken down by weight classification?
# 

# In[26]:


fig = plt.figure(figsize = (20, 10))
fig.suptitle("Number of Respondents per Weight Classification")

#Count the number of datapoints attributed with each weight category
counts = df1["NObeyesdad"].value_counts()
plt.bar(counts.index, counts.values, color="#9381ff")
plt.xlabel("Weight Classification")
plt.ylabel("Number of Respondents")
plt.show()


# ### Graph 15 : What is the average age by weight classification?

# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure that the plot is displayed within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Create the figure and set the super title
fig = plt.figure(figsize=(16, 6))
fig.suptitle("Average Age for Weight Classification and Gender")

# Create the bar plot using Seaborn
sns.barplot(x=df1.NObeyesdad, y=df1.Age, hue=df1.Gender, palette="YlGnBu", order=["Insufficient Weight", "Normal Weight", "Overweight Level I", "Overweight Level II", "Obesity Type I", "Obesity Type II", "Obesity Type III"])

# Set the x-axis and y-axis labels
plt.xlabel("Weight Classification")
plt.ylabel("Age")

# Display the plot
plt.show()


# This graph also implies that females tend to have a higher average age than males by a couple of years for each category besides type II obesity. As the error bars for all of the categories and genders are short, this means that the data did not vary much and does not include a lot of uncertainty.

# ### Graph 16 : How is BMI distributed by weight category?
# 

# In[28]:


plt.figure(figsize=(18,6))
sn.boxplot(x = 'NObeyesdad', y = 'BMI',  
           order=["Insufficient Weight", "Normal Weight", "Overweight Level I", "Overweight Level II","Obesity Type I", "Obesity Type II", "Obesity Type III"], 
           data = df1, palette = "YlGnBu").set_title('BMI and Weight Category')
plt.ylabel("BMI (kg/$cm^2$)", size=12)
plt.xlabel("Weight Category", size=12)


# This graph shows that there is a clear relationship between BMI levels and different weight categories, which helps confirm that BMI was used to characterize weight categories. The medians of each category are separated by similar intervals of around 5 kg/m², although some intervals were smaller than others. There are also not too many outliers, indicating that there is a strong correlation, so BMI can be used as a quantitative variable

# ## Graph 17-20 :Relationship between BMI and various factors

# In[29]:


plt.figure(figsize=(18,12))

#subplot 1: high caloric food 
plt.subplot(2, 2, 1)
sn.boxplot(x = 'FAVC', y = 'BMI', order=["yes", "no"],data = df1, palette = "Set2").set_title('Relationship Between Eating High Caloric Food and BMI')
plt.xlabel("Frequent Consumption of High Caloric Food?", size=12)
plt.ylabel("BMI (kg/$cm^2$)", size=12)

#subplot 2: alcohol consumption
plt.subplot(2, 2, 2)
sn.boxplot(x = 'CALC', y = 'BMI', data = df1, palette = "Set2").set_title('Relationship Between Alcohol Consumption and BMI')
plt.xlabel("Alcohol Consumption", size=12)
plt.ylabel("BMI (kg/$cm^2$)", size=12)

#subplot 3: family history
plt.subplot(2, 2, 3)
sn.boxplot(x = 'family_history_with_overweight', y = 'BMI', data = df1, palette = "Set2").set_title('Relationship Between Family History and BMI')
plt.xlabel("Family History of Obesity?", size=12)
plt.ylabel("BMI (kg/$cm^2$)", size=12)

#subplot 4: gender
plt.subplot(2, 2, 4)
sn.boxplot(x = 'Gender', y = 'BMI', data = df1, palette = "Set2").set_title('Relationship Between Gender and BMI')
plt.xlabel("Gender", size=12)
plt.ylabel("BMI (kg/$cm^2$)", size=12)

plt.show()


# 1. The first subplot illustrates how those that frequently consume high-calorie food have a median BMI higher than the median BMI of those who don’t (around 6 kg/cm² higher). This indicates that calorie count is likely a contributing factor to increased body fat. 
# 
# 2. The second subplot shows that there is little relationship between alcohol consumption and BMI, as those who frequently drink alcohol had the same median BMI as those who do not drink alcohol at all
# 
# 3. The third subplot suggests that a family history of obesity is also a contributing factor, as the median BMI of those with a family history of obesity is around 11 kg/m² higher.
# 
# 4. fourth subplot does not show any relationship between gender and BMI as females and males had the same median BMI; however, based on the quartiles, the BMI of females is more spread out
# 

# ## Graph 21-22

# In[30]:


plt.scatter(df1["Height"], df1["Weight"], alpha = 0.5,color="skyblue")
m, b = np.polyfit(df1["Height"], df1["Weight"], 1)
plt.plot(df1["Height"], m * df1["Height"] + b, color = "red")

plt.xlabel("Height [cm]")
plt.ylabel("Weight [kg]")
plt.title("Correlation between 'Height' and 'Weight'")
plt.show()


# we plots each person's weight and height. The red line shows that there is a positive correlation between them, which means an increase in one variable leads to an increase in the other. In other words, taller people are more likely to weight more.
# 

# In[31]:


sn.lmplot(x="Weight", y="Height", hue="Gender", data=df1,  
          palette=dict(Female="#F72585", Male="#4CC9F0"), height=5, aspect=1.7, x_jitter=.1)
plt.title('Relationship Between Weight and Height by Gender')
plt.xlabel("Weight (kg)")
plt.ylabel("Height (cm)")
plt.show()


#  from the above graph we can see that for both 'male'and 'female' we can see positve correlation .
# 

# ##  Graph 23 : correlation matrix 

# In[32]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Select the columns of interest
columns_of_interest = ['Age', 'Height', 'Weight']
subset_data = df1[columns_of_interest]

# Calculate the correlation matrix
correlation_matrix = subset_data.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# ### Age and Height:
# The correlation coefficient between Age and Height is -0.031321, indicating a weak negative correlation. This means that there is no strong linear relationship between Age and Height.
# 
# ### Age and Weight: 
# The correlation coefficient between Age and Weight is 0.197340, indicating a weak positive correlation. This suggests that there is a slight positive linear relationship between Age and Weight.
# 
# ### Height and Weight: 
# The correlation coefficient between Height and Weight is 0.463734, indicating a moderate positive correlation. This suggests that there is a reasonably positive linear relationship between Height and Weight.
# 
# ### multicollinearity :
# multicollinearity refers to high correlation between independent variables. In this case, the correlation coefficients between Age, Height, and Weight are all relatively low. Although Height and Weight have a moderate positive correlation, it does not necessarily indicate a problematic level of multicollinearity. Multicollinearity is typically a concern when the correlation coefficients are close to 1 or -1.
# 
# Therefore, based on the correlation matrix, there doesn't seem to be a significant issue of multicollinearity among the variables Age, Height, and Weight in our dataset.
# 
# 
# 
# 
# 

# # plot of numerical variables 
# ### Graphs 24-26 : How are the heights and weights of the respondents distributed?
# 

# In[33]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(24, 6))

# Age distribution subplot
plt.subplot(1, 3, 1)
sns.distplot(df1["Age"], color="orange").set_title('Age Distribution')
plt.xlabel("Age")

# Plot mean and median values
age_mean = np.mean(df1["Age"])
age_median = np.median(df1["Age"])
plt.axvline(x=age_mean, color='r', linestyle='--', label='Mean')
plt.axvline(x=age_median, color='b', linestyle='--', label='Median')
plt.legend()

# Height distribution subplot
plt.subplot(1, 3, 2)
sns.distplot(df1["Height"], color="blue").set_title('Height Distribution')
plt.xlabel("Height (cm)")

# Plot mean and median values
height_mean = np.mean(df1["Height"])
height_median = np.median(df1["Height"])
plt.axvline(x=height_mean, color='r', linestyle='--', label='Mean')
plt.axvline(x=height_median, color='b', linestyle='--', label='Median')
plt.legend()

# Weight distribution subplot
plt.subplot(1, 3, 3)
sns.distplot(df1["Weight"], color="green").set_title('Weight Distribution')
plt.xlabel("Weight (kg)")

# Plot mean and median values
weight_mean = np.mean(df1["Weight"])
weight_median = np.median(df1["Weight"])
plt.axvline(x=weight_mean, color='r', linestyle='--', label='Mean')
plt.axvline(x=weight_median, color='b', linestyle='--', label='Median')
plt.legend()

plt.show()


#  #### Age
# The youngest person in the dataset is 14 years old, and the oldest one - 61 years of age. Values in this column are not normally distributed; the historgram is positively skewed with mean (24.35) and median (23) closer to the lower bound.
# 
# #### Height  
# Distribution of height values is plotted below. Most people are 152 cm - 180 cm tall. Both mean and median values are  around 170 cm Still, height values do not seem to be normally distributed.
# 
# #### Weight  
# Distribution is more or less bi-modal; the mean and the median are shifted to the left because of the larger number of people weighting 80 kg. 
# 

# # What are the likely distributions of the numeric variables?  
# Most of our data is categorical with the exception of age, height and weight. We can look at their distribution in our dataset.
# 

# ## Graph 27-29

# In[34]:


import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot

columns_to_plot = ['Age', 'Weight', 'Height']

for c in columns_to_plot:
    plt.figure(figsize=(8, 5))
    fig = qqplot(df1[c], line='45', fit=True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Theoretical quantiles", fontsize=15)
    plt.ylabel("Sample quantiles", fontsize=15)
    plt.title("Q-Q plot of {}".format(c), fontsize=16)
    plt.grid(True)
    plt.show()


# The Q-Q (Quantile-Quantile) plot is a graphical tool used to assess whether a given variable follows a particular theoretical distribution, such as a normal distribution. It compares the quantiles of the variable's observed data against the quantiles of the corresponding theoretical distribution.
# 
# Interpreting the Q-Q plots for each variable:
# - Age: The Q-Q plot for Age doesnt seem to fit the line well 
# - Weight: The Q-Q plot for Weight exhibits a slight deviation from the 45-degree line towards the upper tail, indicating that     the distribution of Weight may have heavier tails compared to a normal distribution.
# - Height: The Q-Q plot for Height shows a slight curvature at the ends, suggesting that the distribution of Height may deviate     from a normal distribution, particularly in the extreme values.
# 

# ## Graph 30-32 :outlier detection 

# In[35]:


import pandas as pd
import matplotlib.pyplot as plt

# Select the variables of interest
df1 = df[['Age', 'Height', 'Weight']]

# Plot boxplots for each variable
fig = plt.figure(figsize=(10, 7))
plt.subplot(1, 3, 1)
plt.boxplot(df1['Age'])
plt.title('Age')
plt.subplot(1, 3, 2)
plt.boxplot(df1['Height'])
plt.title('Height')
plt.subplot(1, 3, 3)
plt.boxplot(df1['Weight'])
plt.title('Weight')
plt.show()


# In[36]:


import pandas as pd

# Read the data from the csv file
df1 = pd.read_csv("obesity clean data kaggle.csv")

# Select the 'Age' variable
age = df1['Age']

# Calculate the first and third quartiles
Q1 = age.quantile(0.25)
Q3 = age.quantile(0.75)

# Print the results
print("First quartile (Q1): ", Q1)
print("Third quartile (Q3): ", Q3)

# Calculate the interquartile range (IQR)
IQR = Q3 - Q1

# Calculate the lower and upper limits using the IQR*1.5 rule
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

# Print the results
print("Interquartile range (IQR): ", IQR)
print("Lower limit: ", lower_limit)
print("Upper limit: ", upper_limit)


# In[37]:


import pandas as pd

# Read the data from the csv file
df1 = pd.read_csv("obesity clean data kaggle.csv")

# Select the 'Weight' variable
weight = df1['Weight']

# Calculate the first and third quartiles
Q1 = weight.quantile(0.25)
Q3 = weight.quantile(0.75)

# Print the results
print("First quartile (Q1): ", Q1)
print("Third quartile (Q3): ", Q3)

# Calculate the interquartile range (IQR)
IQR = Q3 - Q1

# Calculate the lower and upper limits using the IQR*1.5 rule
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Print the results
print("Interquartile range (IQR): ", IQR)
print("Lower limit: ", lower_limit)
print("Upper limit: ", upper_limit)


# In[ ]:





# Considering IQR, I can consider that:
# - for the Age column: values lower than 11.0 and higher than 35.0 being Outliers;
# - for the Weight column: values lower than 2.57 and higher than 170 being Outliers;
# 

# In[38]:


data_iqr = df1.copy()

# Filter rows based on weight and age conditions
data_iqr.drop(data_iqr[(data_iqr.Weight > 170) | (data_iqr.Weight < 2.57)].index, inplace=True)
data_iqr.drop(data_iqr[(data_iqr.Age > 35.0) | (data_iqr.Age < 11.0)].index, inplace=True)

# Display the filtered DataFrame
print(data_iqr)


# In[39]:


len(data_iqr)


# ## Graph 33-34

# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot for Weight
sns.boxplot(y="Weight", data=data_iqr)
plt.yticks(fontsize=10)
plt.ylabel('Weight', fontsize=12)
plt.show()

# Boxplot for Age
sns.boxplot(y="Age", data=data_iqr)
plt.yticks(fontsize=10)
plt.ylabel('Age', fontsize=12)
plt.show()


# no outliers in the data. after removing the outliers the data size reduce to (1950.17)

# In[41]:


len(data_iqr)


# #data size before analysis (2111 ,17)
# 
# #data size after dropping the duplicate  (2065,17)
# 
# #data size after outlier detection (1950 ,17)

# In[42]:


# Create a new DataFrame df2 based on the changes
df2 = data_iqr.copy()

# Replace '_' with ' ' in 'NObeyesdad' and 'MTRANS' columns
df2['NObeyesdad'] = df2['NObeyesdad'].apply(lambda x: x.replace('_', ' '))
df2['MTRANS'] = df2['MTRANS'].apply(lambda x: x.replace('_', ' '))

# Multiply 'Height' by 100 and round to 1 decimal place
df2['Height'] = df2['Height'] * 100
df2['Height'] = df2['Height'].round(1)

# Round 'Weight' and 'Age' columns to 1 decimal place
df2['Weight'] = df2['Weight'].round(1)
df2['Age'] = df2['Age'].round(1)

# Display the first few rows of the new DataFrame df2
print(df2.head())


# In[43]:


df2


# In[44]:


df2.info()


# # Chi square test 

# ## pivot tables (1-13)

# NObeyesdad vs 
# (Gender,
# family_history_with_overweight,
# FAVC,
# FCVC,
# NCP,
# CAEC,
# SMOKE,
# CH2O,
# SCC,
# FAF,
# TUE,
# CALC,
# MTRANS)

# In[45]:


import pandas as pd

# Create pivot table for 'Gender' with filled NaN values and margins
pivot_gender = pd.pivot_table(df2, index='NObeyesdad', columns='Gender', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_gender)
print()

# Create pivot table for 'family_history_with_overweight' with filled NaN values and margins
pivot_family = pd.pivot_table(df2, index='NObeyesdad', columns='family_history_with_overweight', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_family)
print()

# Create pivot table for 'FAVC' with filled NaN values and margins
pivot_favc = pd.pivot_table(df2, index='NObeyesdad', columns='FAVC', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_favc)
print()

# Create pivot table for 'FCVC' with filled NaN values and margins
pivot_fcvc = pd.pivot_table(df2, index='NObeyesdad', columns='FCVC', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_fcvc)
print()

# Create pivot table for 'NCP' with filled NaN values and margins
pivot_ncp = pd.pivot_table(df2, index='NObeyesdad', columns='NCP', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_ncp)
print()

# Create pivot table for 'CAEC' with filled NaN values and margins
pivot_caec = pd.pivot_table(df2, index='NObeyesdad', columns='CAEC', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_caec)
print()

# Create pivot table for 'SMOKE' with filled NaN values and margins
pivot_smoke = pd.pivot_table(df2, index='NObeyesdad', columns='SMOKE', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_smoke)
print()

# Create pivot table for 'CH2O' with filled NaN values and margins
pivot_ch2o = pd.pivot_table(df2, index='NObeyesdad', columns='CH2O', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_ch2o)
print()

# Create pivot table for 'SCC' with filled NaN values and margins
pivot_scc = pd.pivot_table(df2, index='NObeyesdad', columns='SCC', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_scc)
print()

# Pivot table for 'FAF' with filled NaN values and margins
pivot_faf = pd.pivot_table(df2, index='NObeyesdad', columns='FAF', values='Gender', aggfunc='count', fill_value=0, margins=True)
print(pivot_faf)


# Create pivot table for 'TUE' with filled NaN values and margins
pivot_tue = pd.pivot_table(df2, index='NObeyesdad', columns='TUE', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_tue)
print()

# Create pivot table for 'CALC' with filled NaN values and margins
pivot_calc = pd.pivot_table(df2, index='NObeyesdad', columns='CALC', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_calc)
print()

# Create pivot table for 'MTRANS' with filled NaN values and margins
pivot_mtrans = pd.pivot_table(df2, index='NObeyesdad', columns='MTRANS', values='FAF', aggfunc='count', fill_value=0, margins=True)
print(pivot_mtrans)
print()


# # chi squre test 

# #### hypothesis setup 
# 
# Null hypothesis (Ho)  : their is no relation between independent variables and target variables 
# 
# Alternative hypothesis (H1) : their is realation between independent variables and target variables 
#     
# 
# 
# under  assumption of   (i) N>50   (ii) oij>5
# 
# EIJ =(Ri*cj)/N
# 
# where 
#        oij -observe frequecy 
#        
#        EIJ - expected frequecy 
#        
#        Ri -  row total
#        
#        CJ-   column total
#        
#        
#  if HO is true  χ2 follows chi square with (r-1)(c-1)    d.f   
# 
#       χ2 = ∑(Oi – Ei)2/Ei      
#        
#        
# decision criterion with level of significance (alpha) =0.05
# 
# reject HO   if
# 
#                cal chi(x^2) > tab(x^2)[D.F.,aplha]
#                
# DO not reject HO   if
# 
#                cal chi(x^2) < tab(x^2)[D.F.,aplha]
# 

# ### Cramér's V 
# 
# Cramér’s V is an effect size measurement for the chi-square test of independence. It measures how strongly two categorical fields are associated 
# 
# 
# 
# It ranges from 0 to 1 where:
# 
# 
# 0 indicates no association between the two variables.
# 
# 1 indicates a perfect association between the two variables.
# 
# It is calculated as:
# 
# 
# ### Cramer’s V = √(X2/n) / min(c-1, r-1)
# 
# where:
# 
# 
# X2: The Chi-square statistic
# 
# n: Total sample size
# 
# r: Number of rows
# 
# c: Number of columns
# 
# 
# 
# ### To interpret Cramer’s V, the following approach is often used:
# 
# ∈[0.1 ,0.2 ]
# 
# : weak association
# 
# V∈[0.4,0.5]
# 
# : medium association
# 
# V>0.5
# 
# : strong association
# 
# 

# In[46]:


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2

# List of variables to analyze
variables = ['Gender', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'TUE', 'CALC', 'MTRANS']

# Create empty lists to store the results
test_statistics = []
critical_values = []

# Loop through each variable
for variable in variables:
    # Create the contingency table for the variable
    contingency_table = pd.pivot_table(df2, index='NObeyesdad', columns=variable, values='FAF', aggfunc='count').fillna(0)

    # Perform the chi-square test of independence
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    # Calculate the tabulated X^2 value at a given significance level
    alpha = 0.05
    tabulated_chi2 = chi2.ppf(1 - alpha, dof)

    # Append the test statistics and critical values to the respective lists
    test_statistics.append(chi2_stat)
    critical_values.append(tabulated_chi2)

    # Print the results
    print(f"Variable: {variable}")
    print("Chi-square statistic:", chi2_stat)
    print("p-value:", p_value)
    print("Degrees of freedom:", dof)
    print("Expected frequencies:")
    print(expected)

    # Compare the chi-square statistic with the tabulated X^2 value
    if chi2_stat > tabulated_chi2:
        print("Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.")
    else:
        print("Fail to reject the null hypothesis. There is no significant relationship between the variable and 'NObeyesdad'.")

    # Calculate Cramer's V
    n = contingency_table.values.sum()
    phi = np.sqrt(chi2_stat / n)
    r, k = contingency_table.shape
    cramers_v = phi / np.sqrt(min(r - 1, k - 1))

    print("Cramer's V:", cramers_v)
    print()

# Print the test statistics and critical values for each variable
for i, variable in enumerate(variables):
    print(f"Variable: {variable}")
    print("Test statistic (calculated chi-square):", test_statistics[i])
    print("Critical value (tabulated chi-square):", critical_values[i])
    print()


# ### chi sq test for variable name  faf 

# In[47]:


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2

# Pivot table for 'FAF' with filled NaN values and margins
pivot_faf = pd.pivot_table(df2, index='NObeyesdad', columns='FAF', values='Gender', aggfunc='count', fill_value=0, margins=True)
print(pivot_faf)

# Perform the chi-square test of independence
chi2_stat, p_value, dof, expected = chi2_contingency(pivot_faf)

# Print the results
print("Chi-square statistic:", chi2_stat)
print("p-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies:")
print(expected)

# Calculate the tabulated X^2 value at a given significance level
alpha = 0.05
tabulated_chi2 = chi2.ppf(1 - alpha, dof)

# Compare the chi-square statistic with the tabulated X^2 value
if chi2_stat > tabulated_chi2:
    print("Reject the null hypothesis. There is a relationship between 'faf' and 'NObeyesdad'.")
else:
    print("Fail to reject the null hypothesis. There is no significant relationship between 'faf' and 'NObeyesdad'.")

# Calculate Cramer's V
n = pivot_faf.values.sum()
phi = np.sqrt(chi2_stat / n)
r, k = pivot_faf.shape
cramers_v = phi / np.sqrt(min(r - 1, k - 1))

print("Cramer's V:", cramers_v)

# Print the test statistic and critical value
print("Test statistic (calculated chi-square):", chi2_stat)
print("Critical value (tabulated chi-square):", tabulated_chi2)


# ##  all result  
1 .Variable: Gender
Chi-square statistic: 653.0178926649858
p-value: 8.479821385358281e-138
Degrees of freedom: 6
Expected frequencies:
[[131.60871795 139.39128205]
 [135.97948718 144.02051282]
 [137.43641026 145.56358974]
 [130.15179487 137.84820513]
 [156.86205128 166.13794872]
 [131.60871795 139.39128205]
 [123.35282051 130.64717949]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.578689010261495

2 .Variable: family_history_with_overweight
Chi-square statistic: 586.1148185133961
p-value: 2.3047485957073043e-123
Degrees of freedom: 6
Expected frequencies:
[[ 51.69846154 219.30153846]
 [ 53.41538462 226.58461538]
 [ 53.98769231 229.01230769]
 [ 51.12615385 216.87384615]
 [ 61.61846154 261.38153846]
 [ 51.69846154 219.30153846]
 [ 48.45538462 205.54461538]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.5482441990589062

3 .Variable: FAVC
Chi-square statistic: 226.54341937477588
p-value: 4.184101769037894e-46
Degrees of freedom: 6
Expected frequencies:
[[ 31.96410256 239.03589744]
 [ 33.02564103 246.97435897]
 [ 33.37948718 249.62051282]
 [ 31.61025641 236.38974359]
 [ 38.0974359  284.9025641 ]
 [ 31.96410256 239.03589744]
 [ 29.95897436 224.04102564]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.340846171314693

4 .Variable: FCVC
Chi-square statistic: 548.9891078078301
p-value: 8.127770031306322e-110
Degrees of freedom: 12
Expected frequencies:
[[ 14.17538462 125.77179487 131.05282051]
 [ 14.64615385 129.94871795 135.40512821]
 [ 14.80307692 131.34102564 136.85589744]
 [ 14.01846154 124.37948718 129.60205128]
 [ 16.89538462 149.90512821 156.19948718]
 [ 14.17538462 125.77179487 131.05282051]
 [ 13.28615385 117.88205128 122.83179487]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.3751885364712568

5 .Variable: NCP
Chi-square statistic: 476.57757905181785
p-value: 8.680962128293374e-90
Degrees of freedom: 18
Expected frequencies:
[[ 40.02461538  22.37487179 188.72717949  19.87333333]
 [ 41.35384615  23.11794872 194.99487179  20.53333333]
 [ 41.79692308  23.36564103 197.08410256  20.75333333]
 [ 39.58153846  22.12717949 186.63794872  19.65333333]
 [ 47.70461538  26.66820513 224.94051282  23.68666667]
 [ 40.02461538  22.37487179 188.72717949  19.87333333]
 [ 37.51384615  20.97128205 176.88820513  18.62666667]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.28542293676116953

6 .Variable: CAEC
Chi-square statistic: 763.8845983916383
p-value: 1.52825930324393e-150
Degrees of freedom: 18
Expected frequencies:
[[  6.94871795  32.10307692 224.86051282   7.08769231]
 [  7.17948718  33.16923077 232.32820513   7.32307692]
 [  7.25641026  33.52461538 234.8174359    7.40153846]
 [  6.87179487  31.74769231 222.37128205   7.00923077]
 [  8.28205128  38.26307692 268.00717949   8.44769231]
 [  6.94871795  32.10307692 224.86051282   7.08769231]
 [  6.51282051  30.08923077 210.75487179   6.64307692]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.3613565605155529

7 .Variable: SMOKE
Chi-square statistic: 36.47517047364682
p-value: 2.2282319881756146e-06
Degrees of freedom: 6
Expected frequencies:
[[265.58   5.42]
 [274.4    5.6 ]
 [277.34   5.66]
 [262.64   5.36]
 [316.54   6.46]
 [265.58   5.42]
 [248.92   5.08]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.13676701220510445

8 .Variable: CH2O
Chi-square statistic: 161.81197738258012
p-value: 2.244066430114676e-28
Degrees of freedom: 12
Expected frequencies:
[[ 61.14871795 142.44871795  67.4025641 ]
 [ 63.17948718 147.17948718  69.64102564]
 [ 63.85641026 148.75641026  70.38717949]
 [ 60.47179487 140.87179487  66.65641026]
 [ 72.88205128 169.78205128  80.33589744]
 [ 61.14871795 142.44871795  67.4025641 ]
 [ 57.31282051 133.51282051  63.17435897]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.20369155753471468

9. Variable: SCC
Chi-square statistic: 111.62286884615307
p-value: 9.31918761655232e-22
Degrees of freedom: 6
Expected frequencies:
[[257.93641026  13.06358974]
 [266.5025641   13.4974359 ]
 [269.35794872  13.64205128]
 [255.08102564  12.91897436]
 [307.42974359  15.57025641]
 [257.93641026  13.06358974]
 [241.75589744  12.24410256]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.239254042482423

10 .Variable: TUE
Chi-square statistic: 170.28650587836142
p-value: 4.171372303178099e-30
Degrees of freedom: 12
Expected frequencies:
[[114.09794872 123.27025641  33.63179487]
 [117.88717949 127.36410256  34.74871795]
 [119.15025641 128.72871795  35.12102564]
 [112.83487179 121.90564103  33.25948718]
 [135.99128205 146.92358974  40.08512821]
 [114.09794872 123.27025641  33.63179487]
 [106.94051282 115.5374359   31.52205128]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.2089574278064249

11 .Variable: CALC
Chi-square statistic: 341.5149893891767
p-value: 1.3037348657558958e-61
Degrees of freedom: 18
Expected frequencies:
[[1.38974359e-01 7.78256410e+00 1.82195385e+02 8.08830769e+01]
 [1.43589744e-01 8.04102564e+00 1.88246154e+02 8.35692308e+01]
 [1.45128205e-01 8.12717949e+00 1.90263077e+02 8.44646154e+01]
 [1.37435897e-01 7.69641026e+00 1.80178462e+02 7.99876923e+01]
 [1.65641026e-01 9.27589744e+00 2.17155385e+02 9.64030769e+01]
 [1.38974359e-01 7.78256410e+00 1.82195385e+02 8.08830769e+01]
 [1.30256410e-01 7.29435897e+00 1.70766154e+02 7.58092308e+01]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.2416167019574941

12 .Variable: MTRANS
Chi-square statistic: 228.09116304298257
p-value: 3.474956754429281e-35
Degrees of freedom: 24
Expected frequencies:
[[ 42.94307692   0.83384615   1.25076923 218.32871795   7.64358974]
 [ 44.36923077   0.86153846   1.29230769 225.57948718   7.8974359 ]
 [ 44.84461538   0.87076923   1.30615385 227.99641026   7.98205128]
 [ 42.46769231   0.82461538   1.23692308 215.91179487   7.55897436]
 [ 51.18307692   0.99384615   1.49076923 260.22205128   9.11025641]
 [ 42.94307692   0.83384615   1.25076923 218.32871795   7.64358974]
 [ 40.24923077   0.78153846   1.17230769 204.63282051   7.16410256]]
Reject the null hypothesis. There is a relationship between the variable and 'NObeyesdad'.
Cramer's V: 0.17100425959718776


13  variable : FAF
Chi-square statistic: 262.4136645950611
p-value: 6.333848196830645e-40
Degrees of freedom: 28
Expected frequencies:
[[  90.05538462  101.03435897   64.76205128   15.14820513  271.        ]
 [  93.04615385  104.38974359   66.91282051   15.65128205  280.        ]
 [  94.04307692  105.50820513   67.62974359   15.81897436  283.        ]
 [  89.05846154   99.91589744   64.04512821   14.98051282  268.        ]
 [ 107.33538462  120.42102564   77.18871795   18.05487179  323.        ]
 [  90.05538462  101.03435897   64.76205128   15.14820513  271.        ]
 [  84.40615385   94.69641026   60.69948718   14.19794872  254.        ]
 [ 648.          727.          466.          109.         1950.        ]]
Reject the null hypothesis. There is a relationship between 'faf' and 'NObeyesdad'.
Cramer's V: 0.09170983795671449



Variable: Gender
Test statistic (calculated chi-square): 653.0178926649858
Critical value (tabulated chi-square): 12.591587243743977

Variable: family_history_with_overweight
Test statistic (calculated chi-square): 586.1148185133961
Critical value (tabulated chi-square): 12.591587243743977

Variable: FAVC
Test statistic (calculated chi-square): 226.54341937477588
Critical value (tabulated chi-square): 12.591587243743977

Variable: FCVC
Test statistic (calculated chi-square): 548.9891078078301
Critical value (tabulated chi-square): 21.02606981748307

Variable: NCP
Test statistic (calculated chi-square): 476.57757905181785
Critical value (tabulated chi-square): 28.869299430392623

Variable: CAEC
Test statistic (calculated chi-square): 763.8845983916383
Critical value (tabulated chi-square): 28.869299430392623

Variable: SMOKE
Test statistic (calculated chi-square): 36.47517047364682
Critical value (tabulated chi-square): 12.591587243743977

Variable: CH2O
Test statistic (calculated chi-square): 161.81197738258012
Critical value (tabulated chi-square): 21.02606981748307

Variable: SCC
Test statistic (calculated chi-square): 111.62286884615307
Critical value (tabulated chi-square): 12.591587243743977

Variable: TUE
Test statistic (calculated chi-square): 170.28650587836142
Critical value (tabulated chi-square): 21.02606981748307

Variable: CALC
Test statistic (calculated chi-square): 341.5149893891767
Critical value (tabulated chi-square): 28.869299430392623

Variable: MTRANS
Test statistic (calculated chi-square): 228.09116304298257
Critical value (tabulated chi-square): 36.41502850180731

Variable: FAF
Test statistic (calculated chi-square): 262.4136645950611
Critical value (tabulated chi-square): 41.33713815142739


# In[ ]:





# ## summary 

# ### To interpret Cramer’s V, the following approach is often used:
# 
# ∈[0.1 ,0.2 ]
# 
# : weak association
# 
# V∈[0.4,0.5]
# 
# : medium association
# 
# V>0.5
# 
# : strong association
# 
# 

# | Variable    | D.F | Calculated chi square | Tabulated chi square | p-value        | Decision      | Cramers V  | Interpretation of Cramers V |
# |-------------|-----|----------------------|----------------------|----------------|---------------|------------|-----------------------------|
# | Gender      | 6   | 653.017              | 12.59                | 8.47e-138     | Reject Null   | 0.578689   | Strong                      |
# | Family hist | 6   | 586.114              | 12.59                | 2.304e-123    | Reject Null   | 0.5482     | Strong                      |
# | FAVC        | 6   | 226.543              | 12.59                | 4.18e-46      | Reject Null   | 0.340846   | Medium                      |
# | FCVC        | 12  | 548.989              | 21.02                | 8.127e-110    | Reject Null   | 0.3375188  | Medium                      |
# | NCP         | 18  | 476.5775             | 28.86                | 8.680e-90     | Reject Null   | 0.285422   | Weak                        |
# | CAEC        | 18  | 763.8845             | 28.86                | 1.528e-150    | Reject Null   | 0.3613     | Medium                      |
# | SMOKE       | 6   | 36.475717            | 12.59                | 2.22e-06      | Reject Null   | 0.1367     | Weak                        |
# | CH2O        | 12  | 161.811              | 21.02                | 2.44e-28      | Reject Null   | 0.2036     | Weak                        |
# | SCC         | 6   | 111.6228             | 12.59                | 9.319e-22     | Reject Null   | 0.2392     | Weak                        |
# | FAF         | 28  | 262.4136             | 41.33                | 6.333e-40     | Reject Null   | 0.0917     | Weak                        |
# | TUE         | 12  | 170.2865             | 21.02                | 4.171e-30     | Reject Null   | 0.2089     | Weak                        |
# | CALC        | 18  | 341.514              | 28.86                | 1.303e-61     | Reject Null   | 0.2416     | Weak                        |
# | MTRANS      | 24  | 228.09               | 36.415               | 3.474e-35     | Reject Null   | 0.1710     | Weak                        |
# 

# # feature encoding 

#  summary of the encoding methods for each variable, categorized into One-Hot Encoding, Ordinal Encoding, and No Encoding:
# 
# One-Hot Encoding:
# 1. Gender - Encoding Applied
# 2. family_history_with_overweight - Encoding Applie 
# 3. SMOKE - Encoding Applied
# 4. SCC - Encoding Applied
# 5. CALC - Encoding Applied
# 6. MTRANS - Encoding Applied
# 
# Ordinal Encoding:
# 1. CAEC - Encoding Applied
# 2. NObeyesdad - Encoding Applied
# 3. CALC - Encoding Applied
# 
# No Encoding:
# 1. Age - No Encoding
# 2. Height - No Encoding
# 3. Weight - No Encoding
# 4. FAVC
# 5.FCVC
# 6.NCP
# 7.CH20
# 8.FAF
# 9.TUE
# 
# 
# simplified summary of the encoding methods and transformations for all variables:
# (#-prelable data/ no encoding)
# 
# 1. Gender:
#    - Method: One-Hot Encoding
#    - Before encoding: Male/Female
#    - After encoding: Male = 1, Female = 0
# 
# 2. Age:
#    - Method: No encoding (Numeric variable)
# 
# 3. Height:
#    - Method: No encoding (Numeric variable)
# 
# 4. Weight:
#    - Method: No encoding (Numeric variable)
# 
# 5. family_history_with_overweight:
#    - Method: One-Hot Encoding
#    - Before encoding: Yes/No
#    - After encoding: Yes = 1, No = 0
# 
# 6. FAVC:###
#    - Method: One-Hot Encoding
#    - Before encoding: Yes/No
#    - After encoding: Yes = 1, No = 0
# 
# 7. FCVC:###
#    - Method: One-Hot Encoding
#    - Before encoding: 1 = never, 2 = sometimes, 3 = always
#    - After encoding: Always = 3, Sometimes = 2, Never = 1
# 
# 8. NCP: ###
#    - Method: One-Hot Encoding
#    - Before encoding: 1, 2, 3, or 4 meals
#    - After encoding: 4 meals = 4, 3 meals = 3, 2 meals = 2, 1 meal = 1
# 
# 9. CAEC:
#    - Method: Ordinal Encoding
#    - Before encoding: No, Sometimes, Frequently, Always
#    - After encoding: 'No' = 0, 'Sometimes' = 1, 'Frequently' = 2, 'Always' = 3
# 
# 10. SMOKE:
#     - Method: One-Hot Encoding
#     - Before encoding: Yes/No
#     - After encoding: Yes = 1, No = 0
# 
# 11. CH20:###
#     - Method: One-Hot Encoding
#     - Before encoding: 1 = less than a liter, 2 = 1–2 liters, 3 = more than 2 liters
#     - After encoding: More than 2 liters = 3, 1-2 liters = 2, Less than a liter = 1
# 
# 12. SCC:
#     - Method: One-Hot Encoding
#     - Before encoding: Yes/No
#     - After encoding: Yes = 1, No = 0
# 
# 13. FAF:###
#     - Method: One-Hot Encoding
#     - Before encoding: 0 = none, 1 = 1 to 2 days, 2 = 2 to 4 days, 3 = 4 to 5 days
#     - After encoding: 4 to 5 days = 3, 2 to 4 days = 2, 1 to 2 days = 1, None = 0
# 
# 14. TUE:###
#     - Method: One-Hot Encoding
#     - Before encoding: 0 = 0–2 hours, 1 = 3–5 hours, 2 = more than 5 hours
#     - After encoding: More than 5 hours = 2, 3-5 hours = 1, 0-2 hours = 0
# 
# 15. CALC:
#     - Method: One-Hot Encoding
#     - Before encoding: No, Sometimes, Frequently, Always
#     - After encoding: 'No' = 0, 'Sometimes' = 1, 'Frequently' = 2, 'Always' = 3
# 
# 16. MTRANS:
#     - Method: One-Hot Encoding
#     - Before encoding: Public Transportation, Motorbike, Bike,

# ### summary 

#  tabular format:
# 
# | Variable                          | Before Encoding                          | After Encoding                                   | Changes in Ordinal Encoding                                         |
# |-----------------------------------|------------------------------------------|--------------------------------------------------|-------------------------------------------------------------------|
# | Age                               | Numeric variable                         | No Encoding (Numeric variable)                   |                                                                   |
# | Height                            | Numeric variable                         | No Encoding (Numeric variable)                   |                                                                   |
# | Weight                            | Numeric variable                         | No Encoding (Numeric variable)                   |                                                                   |
# | family_history_with_overweight    | Yes/No                                   | One-Hot Encoding: Yes = 1, No = 0                |                                                                   |
# | FAVC                              | Yes/No                                   | One-Hot Encoding: Yes = 1, No = 0                |                                                                   |
# | FCVC                              | Frequency of consumption of vegetables    | No Encoding                                      | 1 = never, 2 = sometimes, 3 = always                               |
# | NCP                               | Number of main meals                      | No Encoding                                      | 1, 2, 3, or 4 meals                                               |
# | CAEC                              | No, Sometimes, Frequently, Always         | Ordinal Encoding                                 | No = 0, Sometimes = 1, Frequently = 2, Always = 3                  |
# | SMOKE                             | Yes/No                                   | One-Hot Encoding: Yes = 1, No = 0                |                                                                   |
# | CH2O                              | Consumption of water daily                | No Encoding                                      | 1 = less than a liter, 2 = 1–2 liters, 3 = more than 2 liters      |
# | SCC                               | Yes/No                                   | One-Hot Encoding: Yes = 1, No = 0                |                                                                   |
# | FAF                               | Physical activity frequency               | No Encoding                                      | 0 = none, 1 = 1 to 2 days, 2 = 2 to 4 days, 3 = 4 to 5 days         |
# | TUE                               | Time using technology devices             | No Encoding                                      | 0 = 0–2 hours, 1 = 3–5 hours, 2 = more than 5 hours                 |
# | CALC                              | No, Sometimes, Frequently, Always         | One-Hot Encoding                                 | No = 0, Sometimes = 1, Frequently = 2, Always = 3                  |
# | MTRANS                            | Public Transportation, Motorbike, Bike    | One-Hot Encoding                                 |                                                                   |
# | NObeyesdad                        | Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III | Ordinal Encoding | Insufficient_Weight = 0, Normal_Weight = 1, Overweight_Level_I = 2, Overweight_Level_II = 3, Obesity_Type_I = 4, Obesity_Type_II = 5, Obesity_Type_III = 6 |

# In[48]:


import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Perform one-hot encoding
one_hot_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'MTRANS']
df_ohe = pd.get_dummies(df2, columns=one_hot_cols, drop_first=True)

# Perform ordinal encoding
ordinal_cols = ['CAEC', 'CALC', 'NObeyesdad']
ordinal_mapping = {
    'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'NObeyesdad': {'Insufficient Weight': 0, 'Normal Weight': 1, 'Overweight Level I': 2,
                   'Overweight Level II': 3, 'Obesity Type I': 4, 'Obesity Type II': 5, 'Obesity Type III': 6}
}

df_ord_enc = df_ohe.copy()
for col in ordinal_cols:
    ord_enc = OrdinalEncoder(categories=[list(ordinal_mapping[col].keys())])
    df_ord_enc[col] = ord_enc.fit_transform(df_ohe[col].values.reshape(-1, 1))

# Print the transformed data
print(df_ord_enc)
df_ord_enc.head()

# Get the column index of "NObeyesdad"
target_column_index = df_ord_enc.columns.get_loc("NObeyesdad")

# Reorder the columns, placing "NObeyesdad" at the last position
cols = list(df_ord_enc.columns)
cols.pop(target_column_index)
cols.append("NObeyesdad")
df_ord_enc = df_ord_enc[cols]
df_ord_enc = df_ord_enc.drop('NObeyesdad', axis=1).join(df_ord_enc['NObeyesdad'])

# Print the modified DataFrame
print(df_ord_enc)


# In[49]:


df_ord_enc.columns


# # standardizartion 

# Standardization:
# Standardization, also known as z-score normalization, is a technique used to transform a distribution of data to have a mean of 0 and a standard deviation of 1. It allows for the comparison of data points from different distributions. The formula for standardizing a data point, x, is:
# 
# z = (x - μ) / σ
# 
# Where:
# - z is the standardized value
# - x is the original data point
# - μ is the mean of the distribution
# - σ is the standard deviation of the distribution
# 
# 

# In[50]:


df_ord_enc


# In[51]:


from sklearn.preprocessing import StandardScaler

# Perform Standardization on age, height, and weight
columns_to_scale = ['Age', 'Height', 'Weight']
standard_scaler = StandardScaler()
df_ord_enc_std = df_ord_enc.copy()  # Create a copy of the original DataFrame

# Standardize the selected columns
df_ord_enc_std[columns_to_scale] = standard_scaler.fit_transform(df_ord_enc_std[columns_to_scale])

# Display the standardized DataFrame
df_ord_enc_std


# In[52]:


df_ord_enc_std.columns 


# ## Graph 36 : before and after applying the standardization 

# In[53]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(26, 14))

# Age distribution subplot before standardization
plt.subplot(2, 3, 1)
sns.distplot(df_ord_enc["Age"], color="orange").set_title('Age Distribution (Before Standardization)')
plt.xlabel("Age")

# Plot mean and median values
age_mean = np.mean(df_ord_enc["Age"])
age_median = np.median(df_ord_enc["Age"])
plt.axvline(x=age_mean, color='r', linestyle='--', label='Mean')
plt.axvline(x=age_median, color='b', linestyle='--', label='Median')
plt.legend()

# Age distribution subplot after standardization
plt.subplot(2, 3, 4)
sns.distplot(df_ord_enc_std["Age"], color="orange").set_title('Age Distribution (After Standardization)')
plt.xlabel("Standardized Age")

# Plot mean and median values
age_mean_std = np.mean(df_ord_enc_std["Age"])
age_median_std = np.median(df_ord_enc_std["Age"])
plt.axvline(x=age_mean_std, color='r', linestyle='--', label='Mean')
plt.axvline(x=age_median_std, color='b', linestyle='--', label='Median')
plt.legend()

# Height distribution subplot before standardization
plt.subplot(2, 3, 2)
sns.distplot(df_ord_enc["Height"], color="blue").set_title('Height Distribution (Before Standardization)')
plt.xlabel("Height (cm)")

# Plot mean and median values
height_mean = np.mean(df_ord_enc["Height"])
height_median = np.median(df_ord_enc["Height"])
plt.axvline(x=height_mean, color='r', linestyle='--', label='Mean')
plt.axvline(x=height_median, color='b', linestyle='--', label='Median')
plt.legend()

# Height distribution subplot after standardization
plt.subplot(2, 3, 5)
sns.distplot(df_ord_enc_std["Height"], color="blue").set_title('Height Distribution (After Standardization)')
plt.xlabel("Standardized Height")

# Plot mean and median values
height_mean_std = np.mean(df_ord_enc_std["Height"])
height_median_std = np.median(df_ord_enc_std["Height"])
plt.axvline(x=height_mean_std, color='r', linestyle='--', label='Mean')
plt.axvline(x=height_median_std, color='b', linestyle='--', label='Median')
plt.legend()

# Weight distribution subplot before standardization
plt.subplot(2, 3, 3)
sns.distplot(df_ord_enc["Weight"], color="green").set_title('Weight Distribution (Before Standardization)')
plt.xlabel("Weight (kg)")

# Plot mean and median values
weight_mean = np.mean(df_ord_enc["Weight"])
weight_median = np.median(df_ord_enc["Weight"])
plt.axvline(x=weight_mean, color='r', linestyle='--', label='Mean')
plt.axvline(x=weight_median, color='b', linestyle='--', label='Median')
plt.legend()

# Weight distribution subplot after standardization
plt.subplot(2, 3, 6)
sns.distplot(df_ord_enc_std["Weight"], color="green").set_title('Weight Distribution (After Standardization)')
plt.xlabel("Standardized Weight")

# Plot mean and median values
weight_mean_std = np.mean(df_ord_enc_std["Weight"])
weight_median_std = np.median(df_ord_enc_std["Weight"])
plt.axvline(x=weight_mean_std, color='r', linestyle='--', label='Mean')
plt.axvline(x=weight_median_std, color='b', linestyle='--', label='Median')
plt.legend()

plt.tight_layout()
plt.show()


# ## train test split new 

# In[54]:


from sklearn.model_selection import train_test_split

# Split the data into features (X) and target variable (y)
X = df_ord_enc_std.drop('NObeyesdad', axis=1)
y = df_ord_enc_std['NObeyesdad']

# Perform train-test split
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the train and test sets
print("X_train_new shape:", X_train_new.shape)
print("X_test_new shape:", X_test_new.shape)
print("y_train_new shape:", y_train_new.shape)
print("y_test_new shape:", y_test_new.shape)


# In[55]:


X_train_new


# In[56]:


X_test_new


# In[57]:


y_train_new


# In[58]:


y_test_new


# --------------------------------------------------------------------------------------------------------------------------------

# # Model building 

# ### model evaluation metrics 
# 
# 
# - True Positive (TP): The number of instances that are correctly predicted as positive by the model.
# 
# - True Negative (TN): The number of instances that are correctly predicted as negative by the model.
# 
# - False Positive (FP): The number of instances that are incorrectly predicted as positive by the model (also known as Type I error).
# 
# - False Negative (FN): The number of instances that are incorrectly predicted as negative by the model (also known as Type II error).
# 
# 
# 
# 
# 
# 1 .True Positive Rate (TPR), also known as Sensitivity, Recall, or Hit Rate:
#    TPR = TP / (TP + FN)
# 
# 
# True Positive Rate (TPR) measures the proportion of positive instances correctly identified by the model out of all actual positive instances. It gives an indication of the model's ability to detect positive cases.
# 
# 
# 2. False Positive Rate (FPR):
#    FPR = FP / (FP + TN)
# 
# False Positive Rate (FPR) measures the proportion of negative instances that are incorrectly identified as positive by the model out of all actual negative instances. It represents the model's propensity to produce false alarms.
# 
# 
# 3 . True Negative Rate (TNR), also known as Specificity:
#    TNR = TN / (FP + TN)
# 
# 
# True Negative Rate (TNR) measures the proportion of negative instances correctly identified by the model out of all actual negative instances. It is the complement of the FPR and indicates the model's ability to correctly identify negative cases.
# 
# 4 False Negative Rate (FNR):
#    FNR = FN / (TP + FN)
# 
# False Negative Rate (FNR) measures the proportion of positive instances that are incorrectly identified as negative by the model out of all actual positive instances. It represents the model's failure to detect positive cases.
# 
# 
# 
# 
# 
# ### Precision, Recall, F1 Score:
# Precision, recall, and F1 score are evaluation metrics used in classification tasks, particularly when dealing with imbalanced datasets. These metrics assess the performance of a classification model by considering different aspects of prediction accuracy:
# 
# - Precision: It measures the proportion of correctly predicted positive instances out of all instances predicted as positive.
#    Precision = TP / (TP + FP)
# 
# - Recall (also known as sensitivity or true positive rate): It measures the proportion of correctly predicted positive instances out of all actual positive instances.
#    Recall = TP / (TP + FN)
# 
# - F1 Score: It is the harmonic mean of precision and recall and provides a balanced measure that considers both metrics.
#    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
# 
# Where:
# - TP is the number of true positive predictions
# - FP is the number of false positive predictions
# - FN is the number of false negative predictions
# 
# These metrics help assess the performance of classification models by considering both the ability to correctly identify positive instances (precision) and the ability to capture all positive instances (recall). The F1 score provides a single metric that balances precision and recall.

# In[ ]:





# we build three model for classification task 
#                     
# ### Ordinal logistic regression model 
# 
# ### Random forest 
# 
# ### Decision tree 
#          

# In[ ]:





# ## ordinal logistic regression model 

# Ordinal logistic regression is a statistical method used to model the relationship between an ordinal dependent variable and one or more independent variables. It is an extension of logistic regression, which is used for binary classification problems.
# 
# In ordinal logistic regression, the dependent variable has three or more ordered categories. The goal is to estimate the relationship between the independent variables and the probabilities of the ordinal categories.
# 
# 

# In[59]:


from mord import LogisticIT
from sklearn.metrics import classification_report

# Convert the target variable to int type
y_train_new = y_train_new.astype(int)

# Create and fit the ordinal logistic regression model
ordreg_model_new = LogisticIT()
ordreg_model_new.fit(X_train_new, y_train_new)

# Convert the test set target variable to int type
y_test_new = y_test_new.astype(int)

# Make predictions on the test set
y_pred_new = ordreg_model_new.predict(X_test_new)

# Print the classification report
report_new = classification_report(y_test_new, y_pred_new)
print("Classification Report:\n", report_new)


# In[60]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create the confusion matrix
cm_new = confusion_matrix(y_test_new, y_pred_new)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Ordinal Logistic Regression")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[61]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# Get predicted probabilities for each class
y_pred_proba = ordreg_model_new.predict_proba(X_test_new)

# Compute the unique classes present in the target variable
classes = np.unique(y_train_new)

# Compute the ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i, c in enumerate(classes):
    y_true_class = (y_test_new == c).astype(int)
    y_pred_class = y_pred_proba[:, i]
    fpr[i], tpr[i], _ = roc_curve(y_true_class, y_pred_class)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Define colormap
cmap = cm.get_cmap('tab10')

# Plot the ROC curve for each class with AUC values
plt.figure(figsize=(8, 6))
for i, c in enumerate(classes):
    plt.plot(fpr[i], tpr[i], color=cmap(i), lw=2,
             label='Class {0} (AUC = {1:.2f})'.format(c, roc_auc[i]))

# Plot the random guess curve
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Ordinal Logistic Regression')
plt.legend(loc='lower right')
plt.show()


# In[ ]:





# ## random forest 

# Random Forest is a machine learning algorithm used for both classification and regression tasks. It combines multiple decision trees and creates an ensemble of trees where each tree is built on a different subset of the training data and uses a random subset of features. The final prediction is determined by aggregating the predictions of individual trees. Random Forests are known for their robustness and ability to handle complex relationships in the data.
# 
# 

# In[62]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create an instance of the Random Forest classifier with adjusted hyperparameters
rf_classifier = RandomForestClassifier()

# Fit the Random Forest model to the training data
rf_classifier.fit(X_train_new, y_train_new)

# Make predictions on the training data
y_train_pred = rf_classifier.predict(X_train_new)

# Calculate the train accuracy
train_accuracy = accuracy_score(y_train_new, y_train_pred)
print("Train Accuracy:", train_accuracy)

# Make predictions on the testing data
y_pred_new = rf_classifier.predict(X_test_new)

# Calculate the test accuracy
test_accuracy = accuracy_score(y_test_new, y_pred_new)
print("Test Accuracy:", test_accuracy)

# Generate the classification report for the testing data
classification_rep = classification_report(y_test_new, y_pred_new)
print("Classification Report:\n", classification_rep)


# In[63]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create the confusion matrix
cm = confusion_matrix(y_test_new, y_pred_new)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[64]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import cm

# Get predicted probabilities for each class
y_pred_proba = rf_classifier.predict_proba(X_test_new)

# Compute the ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(rf_classifier.n_classes_):
    fpr[i], tpr[i], _ = roc_curve((y_test_new == i).astype(int), y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Define colormap
cmap = cm.get_cmap('tab10')

# Plot the ROC curve for each class with AUC values
plt.figure(figsize=(8, 6))
for i in range(rf_classifier.n_classes_):
    plt.plot(fpr[i], tpr[i], color=cmap(i), lw=2,
             label='Class {0} (AUC = {1:.2f})'.format(i, roc_auc[i]))

# Plot the random guess curve
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()


# In[65]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Create an instance of the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=8)

# Fit the Random Forest model to the training data
rf_classifier.fit(X_train_new, y_train_new)

# Get the feature importances from the fitted Random Forest model
importances = rf_classifier.feature_importances_

# Get the column names of the features
feature_names = X_train_new.columns

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance - Random Forest')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ## decision tree 

# decision tree is a predictive model that uses a flowchart-like structure to make decisions or predictions. It represents a set of rules based on features of the data and their corresponding target values. Each internal node represents a test on a particular feature, each branch represents the outcome of the test, and each leaf node represents a prediction or a class label. Decision trees are interpretable and widely used in various machine learning tasks.
# 
# 

# In[66]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create an instance of the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=8)

# Fit the Decision Tree model to the training data
dt_classifier.fit(X_train_new, y_train_new)

# Make predictions on the training data
y_train_pred = dt_classifier.predict(X_train_new)

# Calculate the train accuracy
train_accuracy = accuracy_score(y_train_new, y_train_pred)
print("Train Accuracy:", train_accuracy)

# Make predictions on the testing data
y_pred_new = dt_classifier.predict(X_test_new)

# Calculate the test accuracy
test_accuracy = accuracy_score(y_test_new, y_pred_new)
print("Test Accuracy:", test_accuracy)

# Generate the classification report for the testing data
classification_rep = classification_report(y_test_new, y_pred_new)
print("Classification Report:\n", classification_rep)


# #### confusion matrix 

# In[67]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create the confusion matrix
cm = confusion_matrix(y_test_new, y_pred_new)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[68]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Create an instance of the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=8)

# Fit the Decision Tree model to the training data
dt_classifier.fit(X_train_new, y_train_new)

# Get predicted probabilities for each class
y_pred_proba = dt_classifier.predict_proba(X_test_new)

# Compute the ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(dt_classifier.n_classes_):
    fpr[i], tpr[i], _ = roc_curve((y_test_new == i).astype(int), y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curve for each class with AUC values
plt.figure(figsize=(8, 6))
for i in range(dt_classifier.n_classes_):
    plt.plot(fpr[i], tpr[i], lw=2,
             label='Class {0} (AUC = {1:.2f})'.format(i, roc_auc[i]))

# Plot the random guess curve
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Decision Tree')
plt.legend(loc='lower right')
plt.show()


# In[69]:


import numpy as np
import matplotlib.pyplot as plt

# Get the feature importances from the Decision Tree model
importances = dt_classifier.feature_importances_

# Get the column names of the features
feature_names = X_train_new.columns

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance - Decision Tree')
plt.xticks(rotation=90)
plt.show()


# In[70]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create an instance of the DecisionTreeClassifier without regularization
decision_tree = DecisionTreeClassifier(max_depth=5)

# Train the decision tree on the training data without regularization
decision_tree.fit(X_train_new, y_train_new)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, feature_names=X_train_new.columns, filled=True, fontsize=10)
plt.show()


# In[ ]:





# ####  model performance report  
# 
# ordinal logistic regression model accuracy =89 % 
# 
# random forest model accuracy is =94 % 
# 
# decision tree model accuracy is =93 %
# 
# #note : i have not applied the cv technique to any of my model 
#  
# from roc auc curve we can say that model is good fit . with best model with highest accuracy is random forest model 
# 
# #### The feature importance 
# feature importance calculations show us that age,weight and height had the largest influence on the models above. Other factors that had a large influence on models were the consumption of food before meals, physical activity frequency, consumption of alcohol  and the number of main meals. Factors that were not impactful across the board were smoking, calorie consumption monitoring, , and transportation. 

# ## conclusion 
# In this final report, we assessed the performance of three models: ordinal logistic regression, random forest, and decision tree. The accuracy of the ordinal logistic regression model was 89%, while the random forest and decision tree models achieved accuracies of 94% and 93% respectively. It's worth noting that we didn't use cross-validation techniques for any of these models.
# 
# Based on the ROC AUC curve analysis and overall accuracy , the random forest model showed the best fit with the highest accuracy. This indicates that the random forest model is good at predicting obesity.
# 
# When examining feature importance, we found that age, weight, and height had the most influence on the models. Other factors that had a significant impact included food consumption before meals, frequency of physical activity, alcohol consumption, and the number of main meals. On the other hand, factors like smoking, monitoring calorie intake, and transportation didn't have much effect on the models. It's important to mention that most of the data used in this study (77%) was artificially generated using the Weka tool and the SMOTE filter. Only 23% of the data was directly collected from users through a web platform. Therefore, we can't generalize the results completely to real-world situations.
# 
# Looking ahead, the findings of this research can be beneficial for public health management. They can help identify high-risk groups for obesity and develop effective interventions in the early stages of the disease. It's essential to continue improving and expanding this research to enhance our understanding of obesity and support better interventions in the future.
# 
# 

# # NOTEBOOK ENDS HERE 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## train test split 

# In[121]:


from sklearn.model_selection import train_test_split

# Split the data into features (X) and target variable (y)
X = df_ord_enc_std.drop(target_variable, axis=1)
y = df_ord_enc_std[target_variable]

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the train and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


#  X_train

# X_test

# y_train

# y_test

# In[ ]:





# # here are some basic concept 

# metrices used 
#  #### True Positive (TP) : 
# If the actual classification is positive and the predicted classification is positive (1,1) or (PP), this is called a true positive result because the positive sample was correctly identified by the classifier.
# #### True Negative (TN) : 
# If the actual classification is negative and the predicted classification is negative (0,0) or (NN), this is called a true negative result because the negative sample gets correctly identified by the classifier.
# #### False Positive (FP) | Type I Error : 
# If the actual classification is negative and the predicted classification is positive (0,1) or (NP), this is called a false positive result because the negative sample is incorrectly identified by the classifier as being positive.
# #### False Negative (FN) | Type II Error: 
# If the actual classification is positive and the predicted classification is negative (1,0) or (PN), this is called a false negative result because the positive sample is incorrectly identified by the classifier as being negative.
# #### Sensitivity, Recall, Hit Rate, or True Positive Rate (TPR) : 
#  TPR = TP / Actual Positives = TP / (TP + FN) = 1 - FNR
# #### Specificity, Selectivity or True Negative Rate (TNR) : 
# TNR = TN / Actual Negatives = TN / (TN + FP) = 1 - FPR
# #### Miss Rate or False Negative Rate (FNR) :
# FNR = FN / Actual Positives = FN / (FN + TP) = 1 - TPR
# #### Fall-out or False Positive Rate (FPR) :
# FPR = FP / Actual Negatives = FP / (FP + TN) = 1 - TNR
# 

# In[ ]:





# In[ ]:





# In[ ]:





# # Model building 

# final data set on which we will create the models 

# In[ ]:





# ## baseline model 

# In[ ]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Create a DummyClassifier with a strategy of "most_frequent"
baseline_model = DummyClassifier(strategy="most_frequent")

# Fit the model on the training data
baseline_model.fit(X_train, y_train)

# Make predictions on the test data
baseline_predictions = baseline_model.predict(X_test)

# Calculate the accuracy of the baseline model
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
print("Baseline Accuracy:", baseline_accuracy)


# ## logistic regresssion model 

# In[ ]:





# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the logistic regression model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = logreg_model.predict(X_test_scaled)

# Calculate and print the accuracy
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = logreg_model.score(X_train_scaled, y_train)  # Alternatively, you can use the score method
print("Test Accuracy:", test_accuracy)
print("Train Accuracy:", train_accuracy)

# Print the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# ### Based on the evaluation of the logistic regression baseline model, here are some key performance metrics:
# 
# 1. Accuracy: 0.8881
# 2. Precision: The precision values range from 0.80 to 0.96 for the different classes in the dataset.
# 3. Recall: The recall values range from 0.70 to 1.00 for the different classes in the dataset.
# 4. F1-Score: The F1-scores range from 0.73 to 0.98 for the different classes in the dataset.
# 5. Support: The support values indicate the number of instances in each class, helping to understand the distribution of      
#    instances across different classes in the dataset.
# 

# how to interpretat 
# 
# For class 0:
# 
# TP = 85
# TN = 570
# FP = 11
# FN = 3
# 
# Precision = TP / (TP + FP) = 85 / (85 + 11) = 0.8854
# Recall = TP / (TP + FN) = 85 / (85 + 3) = 0.9659
# F1-Score = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (0.8854 * 0.9659) / (0.8854 + 0.9659) = 0.9239
# 
# The final answer for class 0 is:
# Precision = 0.8854
# Recall = 0.9659
# F1-Score = 0.9239
# 
# 
# True Positive (TP): Out of all instances belonging to class 0 (insufficient weight), the model correctly predicted 85 instances as class 0.
# 
# True Negative (TN): The model correctly identified 570 instances as not belonging to class 0 (insufficient weight).
# 
# False Positive (FP): The model incorrectly predicted 11 instances as class 0 (insufficient weight) when they actually did not belong to this category.
# 
# False Negative (FN): The model missed 3 instances of class 0 (insufficient weight) and predicted them as not belonging to this category.
# 
# Precision: The precision of the model for class 0 is 0.8854, indicating that when the model predicts an instance as class 0 (insufficient weight), it is correct 88.54% of the time.
# 
# Recall: The recall of the model for class 0 is 0.9659, which means that the model successfully identifies 96.59% of the instances belonging to class 0 (insufficient weight) correctly.
# 
# F1-Score: The F1-score for class 0 is 0.9239, which is a combined measure of precision and recall. It provides an overall assessment of the model's performance for class 0 (insufficient weight).
# 
# In summary, for class 0 (insufficient weight), the model has a high precision, indicating a low rate of false positives. It also has a high recall, suggesting that it effectively captures most instances of class 0. 
# 

# ### Graph 37 : confusion matrix for logistic regression 

# In[ ]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - logistic regression ")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Overall, the confusion matrix indicates that the logistic regression model is providing accurate predictions for the given classes .

# --------------------------------------------------------------------------------------------------------------

# ## Multinomial logistic regression 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create and fit the multinomial logistic regression model
logreg_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
logreg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg_model.predict(X_test)

# Calculate and print the accuracy
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = logreg_model.score(X_train, y_train)  # Alternatively, you can use the score method
print("Test Accuracy:", test_accuracy)
print("Train Accuracy:", train_accuracy)

# Print the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# ### Graph 38-39

# In[ ]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Multinomial Logistic Regression")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[ ]:


# Get feature importance
feature_importance_rf = rf.feature_importances_

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(X_train.columns, feature_importance_rf)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance (Multinomial )')
plt.xticks(rotation=90)
plt.show()


# ### Based on the evaluation of the multinomial logistic regression model, the following performance metrics were    observed:
# 
# 1. Accuracy: 0.892 (89.2% accuracy)
# 2. Precision: Ranged from 0.81 to 0.97 for different classes
# 3. Recall: Ranged from 0.73 to 1.00 for different classes
# 4. F1-Score: Ranged from 0.77 to 0.98 for different classes
# 5. Support: Indicates the number of instances in each class, providing insights into class distribution in the dataset.
# 
# Overall, the multinomial logistic  model shows good performance with high accuracy and relatively balanced precision, recall, and F1-scores across different classes. 

# --------------------------------------------------------------------------------------------------------------------------------

# ## random forest 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create an instance of the Random Forest classifier without regularization
random_forest = RandomForestClassifier()

# Train the Random Forest classifier without regularization
random_forest.fit(X_train, y_train)

# Make predictions on the train and test data without regularization
y_train_pred = random_forest.predict(X_train)
y_test_pred = random_forest.predict(X_test)

# Evaluate the accuracy of the model without regularization
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Without Regularization:")
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Create an instance of the Random Forest classifier with regularization
random_forest_reg = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5)

# Train the Random Forest classifier with regularization
random_forest_reg.fit(X_train, y_train)

# Make predictions on the train and test data with regularization
y_train_pred_reg = random_forest_reg.predict(X_train)
y_test_pred_reg = random_forest_reg.predict(X_test)

# Evaluate the accuracy of the model with regularization
train_accuracy_reg = accuracy_score(y_train, y_train_pred_reg)
test_accuracy_reg = accuracy_score(y_test, y_test_pred_reg)
print("With Regularization:")
print("Train Accuracy:", train_accuracy_reg)
print("Test Accuracy:", test_accuracy_reg)


# Print the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# In[ ]:





# In[ ]:





# #### note
# we  see that their is  difference between Test Accuracy: 0.93  and Train Accuracy: 0.9992 is large so model is overfitted so we decided to reglularize the paramters.
# The training accuracy is 1.0, indicating that the model has memorized the training data perfectly. However, the test accuracy is 0.937, which is slightly lower than the training accuracy.
# After applying regularization, the model shows improved generalization performance. The training accuracy decreases to 0.892, indicating that the model is no longer memorizing the training data to the same extent. The test accuracy also decreases to 0.827, but it is closer to the training accuracy compared to the model without regularization.
# therfore we can say that model with regularization shows a better balance between training and test accuracy, indicating that it is less likely to be overfitting compared to the model without regularization.

# ### Graph 40-41 

# In[ ]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Get the feature importance from the trained random forest moddecel
feature_importance_rf = rf_model.feature_importances_

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(X_train.columns, feature_importance_rf)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance (Random Forest)')
plt.xticks(rotation=90)
plt.show()


# ### Based on the evaluation of the random forest model, the following performance metrics were  observed:
# 
# 1. Accuracy overall         : 0.92 (92 % accuracy)
# 2. Precision: Ranged from 0.75 to 0.99 for different classes
# 3. Recall: Ranged from 0.81 to 1.00 for different classes
# 4. F1-Score: Ranged from 0.79 to 0.99 for different classes
# 5. Support: Indicates the number of instances in each class, providing insights into class distribution in the dataset.
# 
# Overall, the random forest model shows good performance with high accuracy and relatively balanced precision, recall, and F1-scores across different classes. 

# --------------------------------------------------------------------------------------------------------------------------------

# ## decision tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create an instance of the DecisionTreeClassifier without regularization
decision_tree = DecisionTreeClassifier()

# Train the decision tree on the training data without regularization
decision_tree.fit(X_train, y_train)

# Make predictions on the train and test data
y_train_pred = decision_tree.predict(X_train)
y_test_pred = decision_tree.predict(X_test)

# Evaluate the accuracy of the model without regularization
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Without Regularization:")
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Create an instance of the DecisionTreeClassifier with regularization
decision_tree_reg = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=4)

# Train the decision tree on the training data with regularization
decision_tree_reg.fit(X_train, y_train)

# Make predictions on the train and test data with regularization
y_train_pred_reg = decision_tree_reg.predict(X_train)
y_test_pred_reg = decision_tree_reg.predict(X_test)

# Evaluate the accuracy of the model with regularization
train_accuracy_reg = accuracy_score(y_train, y_train_pred_reg)
test_accuracy_reg = accuracy_score(y_test, y_test_pred_reg)
print("With Regularization:")
print("Train Accuracy:", train_accuracy_reg)
print("Test Accuracy:", test_accuracy_reg)



# Print the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)



# #### note
# we  see that their is  difference between Test Accuracy: 0.95  and Train Accuracy: 0.99 is large so model is overfitted so we decided to reglularize the paramters.
# The training accuracy is 1.0, indicating that the model has memorized the training data perfectly. However, the test accuracy is 0.95, which is slightly lower than the training accuracy.
# After applying regularization, the model shows improved generalization performance. The training accuracy decreases to 0.83, indicating that the model is no longer memorizing the training data to the same extent. The test accuracy also decreases to 0.85, but it is closer to the training accuracy compared to the model without regularization.
# therfore we can say that model with regularization shows a better balance between training and test accuracy, indicating that it is less likely to be overfitting compared to the model without regularization.

# ### Graph 42-44

# In[ ]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - decision tree")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Get the feature importance from the trained decision tree model
feature_importance_dt = decision_tree.feature_importances_

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(X_train.columns, feature_importance_dt)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance (Decision Tree)')
plt.xticks(rotation=90)
plt.show()


# ### Based on the evaluation of the Decision tree  model, the following performance metrics were  observed:
# 
# 1. Accuracy overall         : 0.92 (92 % accuracy)
# 2. Precision: Ranged from 0.75 to 0.99 for different classes
# 3. Recall: Ranged from 0.81 to 1.00 for different classes
# 4. F1-Score: Ranged from 0.79 to 0.99 for different classes
# 5. Support: Indicates the number of instances in each class, providing insights into class distribution in the dataset.
# 
# Overall, the decision tree  model shows good performance with high accuracy and relatively balanced precision, recall, and F1-scores across different classes. 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create an instance of the DecisionTreeClassifier without regularization
decision_tree = DecisionTreeClassifier(max_depth=5)

# Train the decision tree on the training data without regularization
decision_tree.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(80, 20))
plot_tree(decision_tree, feature_names=X_train.columns, filled=True , fontsize=20)
plt.show()


# # conclusion 
# 
# 
# ### final report model performance 
# logistic regression baseline model aacuracy =88 % 
# 
# multinomial logistic regression model accuracy =89 % 
# 
# random forest model accuracy is =92 % 
# 
# decision tree model accuracy is =92 %
# 
# #note : i have not applied the cv technique to any of my model 
# 
# 
# ### The feature importance 
# feature importance calculations show us that age,weight and height had the largest influence on the models above. Other factors that had a large influence on models were the consumption of food before meals, physical activity frequency, consumption of alcohol  and the number of main meals. Factors that were not impactful across the board were smoking, calorie consumption monitoring, , and transportation. most samples are synthetically generated, i.e., they do not reflect the real world.
#  77% of the data was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through a web platform so we cannot genralize the result .

# --------------------------------------------------------------------------------------------------------------------------------

# In[ ]:




