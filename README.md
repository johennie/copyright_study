# Predicting Litigation Outcomes of Civil Copyright Court Cases

Johennie Helton <br>
August, 2024 <br>

This project provides a study of U.S. court outcome and opinions about fair use and copyright
civil cases by analyzing a dataset containing various features related to court cases, 
including text, numerical, and categorical data, building and evaluating
machine learning models to determine their predictive performance. The target variable is the case 
outcome (based on the case opinion), and two sets of features were
tested: one with case opinions (w_op) and one without case opinions (wo_op). <br>

Notebook: https://github.com/johennie/copyright_study/blob/aead29c11eec44e49d0feda0691ddcd9e3e155f1/notebooks/capstone_copyright_case_study.ipynb <br>
Data: https://github.com/johennie/copyright_study/blob/aead29c11eec44e49d0feda0691ddcd9e3e155f1/data/fairuse_copyright_dataset.csv <br>
Script to build dataset: https://github.com/johennie/copyright_study/blob/7858b583524cc639fcf9ec05f57ac3d71a09debc/notebooks/build_dataset.ipynb <br>

## Project Objectives
1.	Model Comparison <br>Evaluate the performance of different machine learning models on the dataset.
2.	Feature Analysis <br>Determine the impact of including case opinions (textual data) on model performance.
3.	Hyperparameter Tuning <br>Optimize model parameters using Grid Search to achieve better results.
 
## Findings

### Model Performance
1. Random Forest <br>Showed signs of overfitting, with perfect training accuracy but lower test accuracy. 
Including text data did not significantly improve performance. 
2. Logistic Regression <br>Benefited from the inclusion of text data, showing improved test accuracy and F1 score
with text data. 
3. KNN<br>Performed poorly overall, with low test accuracy and F1 scores. The inclusion of text data provided 
marginal improvements. 
4. Gradient Boosting <br>Demonstrated good performance, particularly with text data, achieving high test accuracy 
and F1 scores. 
5. SVM <br>Showed improved performance with text data, achieving high test accuracy and F1 scores. 
<br>
### Hyperparameter Tuning
1. Grid Search significantly improved the performance of Logistic Regression, KNN, and Gradient Boosting models. 
2. Optimized models, especially with text data, showed better test accuracy and F1 scores compared to default parameters. 

### Impact of Text Data
1. Including textual case opinions generally improved model performance, particularly for Logistic Regression, SVM, and Gradient Boosting models. 
2. Proper text preprocessing and feature engineering are crucial for leveraging textual data effectively.

### Conclusion
This project demonstrated that incorporating textual data from case opinions can enhance the predictive performance 
of machine learning models for legal case outcomes. However, the benefits vary across different models and depend 
heavily on appropriate text preprocessing and feature engineering. Gradient Boosting and SVM models showed the most 
significant improvements with text data, indicating their ability to effectively utilize additional information.
The Random Forest Classifier with hyperparameter tuning the best model for the set without the opinion (text) 
feature hyperparameter-tuned version (GridSearch with RandomForestClassifier_wo_op) showed more balanced performance, 
suggesting that hyperparameter tuning effectively managed overfitting. Which
indicates that it generalizes well to unseen data and effectively balances precision and recall, making it the most suitable choice among the evaluated models for this specific feature set
Hyperparameter tuning further improved model performance, highlighting the importance of optimizing model parameters. The trade-off between computational cost (fit time) and model performance should be considered when deploying these models in practice.


### Recomendation
Gradient Boosting and SVM models performed better for tasks involving the textual data,
as they demonstrated superior performance and performing hyperparameter tuning using techniques like Grid Search allows us
to achieve the best possible model performance.
<br><br>
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
The dataset we are working with is collected from Harvard Case Law data https://case.law/; reference the script
to build the dataset https://github.com/johennie/copyright_study/blob/7858b583524cc639fcf9ec05f57ac3d71a09debc/notebooks/build_dataset.ipynb <br>
<br>It contains 1043 entries with 7 features and contains the following columns: <br><br>
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
<br> <br>
The dataset is distributed between fair use and copyright case types as shown in the following pie plot.
<br>

![case_type.png](images%2Fcase_type.png)
<br><br>
In this dataset copyright case types may have an outcome of either 'copyright infringement found', 'copyright infringement not found', or 'pending.' Similarly, fair use case types may have an outcome of either 'pending', 'fair use not found', or 'fair use found'.
This is depicted in the following plot, which shows the percentage of each outcome type. 
<br><br>

![outcome.png](images%2Foutcome.png)
<br><br>
The count plot bellow shows the distribution of cases across the different appellate jurisdictions within the federal court system. 
It demonstrates that most of the cases in this dataset are in SCOTUS, then the 2nd Circuit (New York City's lower Manhattan) 
followed by the 9th Circuit (jurisdiction over several districts, including the District of Alaska, Arizona, California, Guam, Hawaii, Idaho, Montana, Nevada, Oregon, and Washington).
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
# 3. Model Evaluation and Analysis

## Methodology
1. Identify the models to fit and test with and use GridSearch for hyperparameter tuning with the aim to identify 
the best-performing algorithm for predicting legal case outcomes based on the given features
<br>RandomForestClassifier: a powerful ensemble method that combines the predictions of multiple decision trees to improve generalization and reduce overfitting.
<br>LogisticRegression: a simple yet effective linear model and as a baseline model due to its fast training times 
<br>KNN: can be effective for smaller datasets and can capture non-linear relationships without the need for explicit model training.
<br>Gradient Boosting: builds models sequentially, with each new model attempting to correct the errors of the previous ones.
<br>SVM: versatile and robust model that can perform both linear and non-linear classification
2. Create two different sets of data 
<br>X with case opinions: with text, court, jurisdiction, tags, year, and case_type_encoded fetaures.
<br>X without case opinions: with court, jurisdiction, tags, year, and case_type_encoded fetaures excluding the text feature.
3. Create two different preprocessors
<br>preprocessor: to handle data with the text (case opinion) feature, as well as numerical and categorical features.
<br>preprocessor_wo_op: to handle numerical and categorical features.
4. Train and test each of the models without case opinions. 
5. Train and test each of the models with case opinions.
<br>
## Fit times and metrics

![results.png](images%2Fresults.png)
<br>
Bellow we visualize both the fit time (in seconds) and the test accuracy for each 
model. The dark green bars represent the fit time (in seconds) for each model, with text 
labels on top showing the values obtained; and the cornflower blue line with markers 
represents the test accuracy for each model. Including case opinions incur higher fit 
times with increased test accuracy especially for Gradient Boosting and SVM models. 
The GridSearch models, especially with text data, show the highest fit times due to 
the hyperparameter tuning process.
<br>

![times_and_testaccuracy.png](images%2Ftimes_and_testaccuracy.png)
<br>
<br><br>
Finally we visualize both the fit time (in seconds) and the F1 score for each model. The dark 
green bars represent the fit time (in seconds) for each model, with text labels on top showing the 
values obtained; and the cornflower blue line with markers represents the F1 score for each model. 
Including case opinions incur in higher fit times but with improved F1 scores for several models, 
particularly Gradient Boosting and SVM. This indicates that while computational costs are higher, 
the models benefit from the additional information provided by the opinion text which leads to better
performance while balancing precision and recall.<br>

![times_and_f1scores.png](images%2Ftimes_and_f1scores.png)
<br>
As we discussed in the findings section above, including case opinions (text feature) generally 
improved the test accuracy and F1 score across most models. The Gradient Boosting model with case opinions achieved the 
highest test accuracy (0.649) and F1 score (0.643) suggesting that this model is well-suited for the dataset,
and GridSearch improved model performance, particularly for Gradient Boosting and SVM models. 
For instance, tuning the SVM model resulted in a test accuracy increase from 0.562 (default) to 0.617, 
and the F1 score rose from 0.560 to 0.612.
<br><br>
For future work, exploring additional text preprocessing techniques to further enhance model 
performance would probably improve results. 
Additionally, expanding the dataset with more cases could provide more robust and generalized predictions.