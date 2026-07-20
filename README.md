Here is a clean, modern, and comprehensive README.md written in English tailored to your Titanic machine learning project.

It highlights the full workflow present in your notebook, from exploratory data analysis (EDA) and robust outlier detection to the comparison of two different feature engineering approaches using different missing value strategies, and finally, model evaluation.

Titanic - Survival Prediction & Advanced Feature Engineering
A comprehensive Machine Learning workflow on the Titanic dataset, focusing heavily on Exploratory Data Analysis (EDA), advanced missing value imputation (KNN vs. Median), custom feature engineering, and robust model comparison.

🚀 Project Overview
The goal of this project is to build a predictive model that determines whether a passenger survived the Titanic disaster. Instead of jumping straight into modeling, this repository demonstrates a thorough Data Science pipeline by addressing missing data via multiple strategies and creating domain-specific features to maximize the predictive power of Random Forest and Logistic Regression classifiers.

📊 Workflow & Features
1. Exploratory Data Analysis (EDA) & Outlier Detection
Data Summary: Analyzed the dataset structure, data types, and missing value percentages (data.info() & data.isna().sum()).

Advanced Outlier Detection: Implemented a custom Interquartile Range (IQR) function (detect_outliers_iqr) to statistically quantify outliers across numerical variables (Age, Fare, SibSp, Parch).

Visualizations: Generated professional box plots and scatter plots (e.g., Age vs Fare) mapping passenger distribution against survival status.

2. Missing Value Imputation Strategies
To evaluate the impact of data imputation on model performance, two distinct datasets were generated:

Strategy A (Median Imputation): Numerical variables were filled with the median, and categorical variables with the mode.

Strategy B (KNN Imputation): Missing values in Age and Fare were imputed using a K-Nearest Neighbors Imputer (n_neighbors=5), predicting missingness based on underlying passenger similarities.

3. Advanced Feature Engineering
Custom features were engineered extracted directly from raw variables to capture hidden socio-economic signals:

new_AgeGroup: Categorized passengers into age segments (Child, Teen, Young, Adult, Older).

new_ageSex: Combined demographic vectors (e.g., Child_female, Adult_male).

new_cabinCount & new_hasCabin: Captured socio-economic status by counting or verifying cabin ownership.

title: Extracted prefixes (Mr, Mrs, Miss, Master, etc.) from the Name column via regex.

new_TravelWithGroup: Grouped passengers by Ticket numbers to flag whether they travelled in groups.

new_familySize & new_isAlone: Calculated relational density on board.

deck: Isolated the specific deck level from the Cabin string.

4. Data Visualization Insights
Generated and saved statistical correlation heatmaps, target distributions, and stacked group trends:

Class vs. Cabin: Analyzed why class-3 passengers mostly lacked recorded cabins.

Deck vs. Survival: Visualized survival rates distributed across different structural decks of the ship.

🛠️ Tech Stack & Libraries
Language: Python 3.x

Data Manipulation: pandas, numpy

Data Visualization: matplotlib, seaborn

Machine Learning: scikit-learn

Imputation: KNNImputer

Preprocessing: StandardScaler, LabelEncoder

Models: RandomForestClassifier, LogisticRegression

Metrics: accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

📈 Key Visualizations Generated
The script automatically dumps high-quality analytical charts into the working directory:

age_vs_fare_scatter.png: Distribution of survivors across age and fare matrix.

pclass_survival_rate.png: Survival rates depending on Passenger Class.

correlation_heatmap.png: Pearson correlation matrix of the engineered numeric features.

deck_survival_rate.png: Survival rate per Deck/Güverte.

💻 How to Run
Clone the repository:

Bash
git clone https://github.com/ecenurbayraktar/titanic_dataset.git
cd titanic-feature-engineering
Install dependencies:

Bash
pip install pandas numpy seaborn matplotlib scikit-learn
Ensure titanic.csv is in the root folder and run the notebook/script.

📌 Conclusions & Future Work
Compare the final metrics (F1-Score and Accuracy) of the models trained with KNN-imputed data against Median-imputed data.

Further optimize hyper-parameters for the RandomForestClassifier using GridSearch.
