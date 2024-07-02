# Predicting Litigation Outcomes of Civil Copyright Court Cases

Johennie Helton <br>
July, 2024 <br>

This project provides a study of U.S. court outcome and opinions about fair use and copyright
civil cases by analyzing a dataset containing various features related to court cases, 
including text data, numerical data, and categorical data, building and evaluating
machine learning models to determine their predictive performance. <br>

Notebook: https://github.com/johennie/copyright_study/blob/aead29c11eec44e49d0feda0691ddcd9e3e155f1/notebooks/capstone_copyright_case_study.ipynb <br>
Data: https://github.com/johennie/copyright_study/blob/aead29c11eec44e49d0feda0691ddcd9e3e155f1/data/fairuse_copyright_dataset.csv <br>

# 1. Business understanding 
The U.S. Copyright Office defines copyright [https://www.copyright.gov/what-is-copyright/]  as a "type of intellectual property that protects original works of authorship as soon as an author fixes the work in a tangible form of expression. In copyright law, there are a lot of different types of works, including paintings, photographs, illustrations, musical compositions, sound recordings, computer programs, books, poems, blog posts, movies, architectural works, plays," etc.. In addition, copyright infrigement involves the unauthorized use or reproduction, distribution, public display, creation or derivative of copyrighted works. <br>
<br>
However, there is no such thing as “global copyright laws.” Though, in some instances the use of copyrighted works is deemed 'fair use,' such as: parody, news reporting, commentary, impact on the market/public, and others. Litigation may occur among the parties owning the copyright and the parties that used or referenced the work.  
<br>
This system analyzes copyright infringement and copyright fair use legal cases brought in the U.S. courts and their outcome. It is important to understand fair use and copyright infringement to ensure the rights of the copyright holders are sustained while balancing public discurse and use for creative works. <br>



## Overview
This analysis evaluates the predictive power of various machine learning models on a subset of civil cases, and focus on the litigation results of copyright infringement found (or not found) and fair use found (or not found) caseslaw.<br>

At first I planned to use the Federal Judicial Center Civil Cases terminated between 2014 and 2023 (with 3557752 rows and 46 columns), and also with the fair use cases (with two datsets of 251 rows and 7 columns, and 251 rows and 9 columns). However, there was no obvious way to associate them with their corresponding opinions and categories (tags). Instead, I created a dataset file with copyright cases from Harvard Case Law (https://case.law/).


# 2. Data Understanding
The dataset we are working with is collected from Harvard Case Law data https://case.law/ and contains 1043 entries with 7 features and contains the following columns: <br><br>
case_type_encoded - the type of case, one of: 'copyright' (1) or 'fair use' (0) <br><br>
year - the year in which the decision or opinion was published or the case was disposed of <br><br>
court - federal or state court. Examples: 'Supreme Court of the United States', 'United States District Court for the District of Massachusetts', etc... <br><br>
jurisdiction - the appellate jurisdiction within the  federal court system of the United States, one of: '2nd Circuit', '9th Circuit', 'Federal Circuit', '11th Circuit', '3rd Circuit', '5th Circuit', '6th Circuit', '4th Circuit', '10th Circuit', '8th Circuit', 'U.S. Supreme Court', '7th Circuit', 'District of Columbia Circuit', '1st Circuit', '-8' (where -8 is unknown)<br><br>
tags - a list of categories associated with the case. Examples: education, film, parody etc... <br><br>
text - the opinion or case summary associated with the case<br><br>
outcome - this is the target, the outcome of the case based on the case's disposition, one of: 'copyright infringement found', 'copyright infringement not found', 'pending', 'fair use not found', 'fair use found' <br><br>

## Exploratory Data Analysis (EDA)
<br>
In this section we aim to understand the data, features and data types as well as potential patterns and relationships. 
We look into percentages of case types, outcomes, and distribution of cases across jurisdictions, courts and tags.
<br>
The dataset is balanced between fair use and copyright case types as shown in the following pie plot.
<br>

![case_type.png](images%2Fcase_type.png)
<br><br>
This plot shows the percentage of each outcome type of 'copyright infringement found', 'copyright infringement not found', 'pending', 'fair use not found', and 'fair use found'. Be aware that in this dataset copyright case types may have an outcome of eithe 'copyright infringement found', 'copyright infringement not found', or 'pending.' Similarly, fair use case types may have an outcome of either 'pending', 'fair use not found', or 'fair use found'.
<br>

![outcome.png](images%2Foutcome.png)
<br><br>
This count plot shows the distribution of cases across the different appellate jurisdictions within the federal court system. It demonstrates that most of the cases in this dataset are in SCOTUS, then the 2nd Circuit followed by the 9th Circuit.
<br>

![jurisdiction.png](images%2Fjurisdiction.png)
<br><br>
Most of the cases by jurisdiction in this dataset are in 'copyright infringement found' cases.
<br>

![case outcome by jurisdiction.png](images%2Fcase%20outcome%20by%20jurisdiction.png)
<br><br>
Bellow we are able to observe that 'research', 'music', and 'film' are the most common tags/categories in this dataset.
<br>

![common_tags.png](images%2Fcommon_tags.png)
<br><br>
Looking at the Number of cases per year shows that cases had been on an increase trend since 1975 but lately they have been on a decline.
<br>

![cases_per_year.png](images%2Fcases_per_year.png)
<br><br>
Finally, we look at a scatter plot of court and jurisdiction with outcome as a hue to identify that SCOTUS has a larger number of pending cases. In addition, most courts have cases unassigned to a jurisdiction.
<br>

![court_jurisdiction_with_outcome.png](images%2Fcourt_jurisdiction_with_outcome.png)
<br><br>
## Data Preprocessing
Several preprocessing steps are applied to ensure the data is ready for modeling:<br>
1.	Handle missing values: 
Missing values in numerical columns are imputed with the mean, 
while missing values in categorical columns were filled with a placeholder 'missing'.<br>
2.	Scale numerical features: 
Numerical features are scaled using StandardScaler to ensure they have a mean of 0 
and a standard deviation of 1. <br>
3.	Encode Categorical Features: 
Categorical features are encoded using OneHotEncoder to convert them 
into a format suitable for machine learning models. <br>
4.	Text Processing: The TfidfVectorizer is used to convert the text data in 
the text column into numerical features, limited to the top 1000 features for simplicity.<br>
5.	Tags Processing: Process the tags column which contains comma-separated strings<br>
<br>

<br><br><br>