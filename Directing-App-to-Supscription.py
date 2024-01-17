import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


## Data ##
df = pd.read_csv(".datasets/appdata10.csv")
top_screens_df = pd.read_csv(".datasets/top_screens.csv").top_screens.values


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1500)


## EDA ##
def check_df(dataframe, head=5):
    print("#################### Shape ####################")
    print(dataframe.shape)
    print("#################### Types ####################")
    print(dataframe.dtypes)
    print("#################### Num of Unique ####################")
    print(dataframe.nunique())  # "dataframe.nunique(dropna=False)" yazarsak null'larıda veriyor.
    print("#################### Head ####################")
    print(dataframe.head(head))
    print("#################### Tail ####################")
    print(dataframe.tail(head))
    print("#################### NA ####################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ####################")
    print(dataframe.describe([0.01, 0.05, 0.75, 0.90, 0.95, 0.99]).T)


check_df(df)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df, True)
import missingno as msno
msno.bar(df)
df["enrolled_date"].fillna(0, inplace=True)  # 1 is member, 0 is a potential member
missing_values_table(df, True)  # checked!
msno.bar(df)  # checked!

df["hour"] = df["hour"].str.slice(1, 3).astype(int)
date_cols = ["first_open", "enrolled_date"]
df[date_cols] = df[date_cols].apply(pd.to_datetime)


def grab_col_names(dataframe, cat_th=16, car_th=24):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col != "user"]
num_cols = [col for col in num_cols if col != "first_open"]
num_cols = [col for col in num_cols if col != "enrolled_date"]

df.dtypes

df["age"].max()  # 101, suspicious!?!
ages = df['age'].value_counts().sort_index()
ages_df = pd.DataFrame(ages)
total_count = ages.sum()
age_ratios = ages / total_count
age_distribution_df = pd.DataFrame({'Age': ages.index, 'Count': ages.values, 'Ratio': age_ratios.values})
df_max_71 = df.copy()
df_max_71 = df_max_71[(df_max_71["age"] < 72)]  # due to low the ratio (lower than 0,001), I'll remove the users who are more than 72.
check_df(df_max_71)

plt.figure(figsize=(10, 6))
sns.histplot(df_max_71['age'], bins=20, kde=False, color='#FF00FF', edgecolor='black')
plt.title('Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if (dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)] != pd.Timestamp(0)).any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df_max_71, col))


### Histograms
def num_summary(dataframe):
    fig, axes = plt.subplots(len(num_cols), 2)
    plt.subplots_adjust(left=0.1, right=0.9, hspace=0.5)

    for i, col in enumerate(num_cols):
        sns.histplot(data=dataframe, x=col, kde=True, ax=axes[i, 0], color='#FF00FF')
        axes[i, 0].set_title(f'{col} Histogram')

        sns.boxplot(x=dataframe[col], ax=axes[i, 1], color='#FF00FF')
        axes[i, 1].set_title(f'{col} Boxplot')

    # plt.tight_layout()
    plt.show(block=True)


num_summary(df_max_71)

filtered_data = df_max_71[df_max_71["enrolled_date"] > "2013-01-01"]  # to remove the rows that were filled 0 before
sns.histplot(filtered_data["enrolled_date"], bins=20, kde=False, color='#FF00FF', edgecolor='black')
plt.title('Enrolled Date - Histogram')
plt.xlabel("Enrolled date")
plt.ylabel('Frequency')
plt.show()


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df_max_71, "numscreens")
check_outlier(df_max_71, "numscreens")


def cat_summary(dataframe, col_name, bar=False, pie=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if dataframe[col_name].dtype == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

    if bar or pie:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        if bar:
            sns.countplot(x=dataframe[col_name], data=dataframe, ax=axes[0], color='#FF00FF')
        if pie:
            col_name_counts = dataframe[col_name].value_counts()
            data = col_name_counts.values
            keys = col_name_counts.keys().values
            colors = sns.color_palette("turbo", len(keys))
            axes[1].pie(data, labels=keys, autopct='%.0f%%', colors=colors)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df_max_71, col, bar=True, pie=True)
# "In the case of imbalanced distribution in your target variable (%62-%37), the accuracy metric can be misleading.
# This is because a model might achieve high accuracy by simply predicting the more prevalent class. In such a scenario,
# focusing on more specific performance metrics like ROC-AUC would be more reliable."



### Correlation with Response Variable
# df_max_71["enrolled"] = pd.to_numeric(df_max_71["enrolled"], errors='coerce')  # errors='coerce', makes non-countable to Nan
colors = sns.color_palette("turbo", len(cat_cols + num_cols))
# other color palette: 'deep' 'muted' 'bright' 'pastel' 'dark' 'colorblind' 'viridis' 'plasma' 'inferno' 'magma'
# 'cividis' 'rocket' 'turbo'
numeric_cols = ["hour", "age", "numscreens"]
df2 = df_max_71[cat_cols + numeric_cols].drop("enrolled", axis=1)
df2.corrwith(df_max_71["enrolled"]).plot.bar(figsize=(20, 10),
                                             title='Correlation with Reposnse variable',
                                             fontsize=15,
                                             rot=45,
                                             grid=True,
                                             color=colors)


### Correlation Matrix
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Generate a mask for the upper triangle
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]

    # 0.90'dan büyük korelasyon değerlerini corr matrisinden kaldır
    for col in drop_list:
        for row in corr.index:
            if corr.loc[row, col] > 0.90:
                corr.loc[row, col] = 0

    if plot:
        sns.set(rc={"figure.figsize": (9, 9)})
        sns.heatmap(corr,
                    mask=mask,
                    center=0,
                    annot=True,
                    square=True,
                    linewidths=.5,
                    fmt=".2f",
                    cmap="magma")
        plt.show(block=True)
    return corr, print(f"Drop List: {drop_list}")


high_correlated_cols(df2, plot=True)


## Feature Engineering ##
df_max_71["difference"] = (df_max_71["enrolled_date"] - df_max_71["first_open"]).dt.total_seconds() / 60  # minute
df_max_71["difference"] = df_max_71["difference"].apply(lambda x: max(x, 0))
df_max_71["difference"].describe([0.01, 0.05, 0.75, 0.90, 0.95, 0.99])

plt.hist(df_max_71["difference"].dropna(), color='#FF00FF', range=[0, 80])
plt.tight_layout()
plt.show()

# Mapping Screens to Fields
df_max_71["screen_list"] = df_max_71["screen_list"].astype(str) + ','

for sc in top_screens_df:
    df_max_71[sc] = df_max_71["screen_list"].str.contains(sc).astype(int)
    df_max_71['screen_list'] = df_max_71["screen_list"].str.replace(sc+",", "")

df_max_71['OTHER_SCREEN_NUM'] = df_max_71['screen_list'].str.count(",")
df_max_71 = df_max_71.drop(columns=['screen_list'])

# Funnels
savings_screens = ["Saving1",
                   "Saving2",
                   "Saving2Amount",
                   "Saving4",
                   "Saving5",
                   "Saving6",
                   "Saving7",
                   "Saving8",
                   "Saving9",
                   "Saving10"]
df_max_71["SAVING_SC"] = df_max_71[savings_screens].sum(axis=1)
df_max_71 = df_max_71.drop(columns=savings_screens)

cm_screens = ["Credit1",
              "Credit2",
              "Credit3",
              "Credit3Container",
              "Credit3Dashboard"]
df_max_71["CREDIT_SC"] = df_max_71[cm_screens].sum(axis=1)
df_max_71 = df_max_71.drop(columns=cm_screens)

cc_screens = ["CC1",
              "CC1Category",
              "CC3"]
df_max_71["CC_SC"] = df_max_71[cc_screens].sum(axis=1)
df_max_71 = df_max_71.drop(columns=cc_screens)

loan_screens = ["Loan",
                "Loan2",
                "Loan3",
                "Loan4"]
df_max_71["LOAN_SC"] = df_max_71[loan_screens].sum(axis=1)
df_max_71 = df_max_71.drop(columns=loan_screens)

cat_cols71, num_cols71, cat_but_car71 = grab_col_names(df_max_71)
check_df(df_max_71)


"""
age.min = 16
age.max = 71

numscreens.min = 1
numscreens.max = 181

first_open.min = 2012-11-23 00:10:19.912000
first_open.min = 2013-07-09 15:52:39.983000

"""
df_max_71['WEEK_OF_YEAR'] = df_max_71['first_open'].dt.isocalendar().week.astype(int)

df_max_71.loc[(df_max_71["age"] >= 16) & (df_max_71["age"] < 25), "SEGMENT_AGE"] = "student"
df_max_71.loc[(df_max_71["age"] >= 25) & (df_max_71["age"] < 35), "SEGMENT_AGE"] = "young"
df_max_71.loc[(df_max_71["age"] >= 35) & (df_max_71["age"] < 45), "SEGMENT_AGE"] = "young-mid"
df_max_71.loc[(df_max_71["age"] >= 45) & (df_max_71["age"] < 55), "SEGMENT_AGE"] = "mid"
df_max_71.loc[(df_max_71["age"] >= 55) & (df_max_71["age"] < 65), "SEGMENT_AGE"] = "mid-old"
df_max_71.loc[df_max_71["age"] >= 65, "SEGMENT_AGE"] = "old"

df_max_71["SEGMENT_SCREEN_NUM"] = pd.cut(x=df_max_71['numscreens'], \
                                         bins=[0, 10, 20, 40, 80, 200], \
                                         labels=["<10", "10-20", "20-40", "40-80", "80-200"])


def create_combined_segment(df, segment1, segment2, new_variable_name):
    df[new_variable_name] = ''

    for category1 in df[segment1].unique():
        for category2 in df[segment2].unique():
            mask = (df[segment1] == category1) & (df[segment2] == category2)
            df.loc[mask, new_variable_name] = f'{category1}_{category2}'

    return df


create_combined_segment(df_max_71, "SEGMENT_AGE", "SEGMENT_SCREEN_NUM", "SEGMENT_AGE-SCREEN_NUM")
create_combined_segment(df_max_71, "SEGMENT_AGE", "WEEK_OF_YEAR", "SEGMENT_AGE-WEEK")
create_combined_segment(df_max_71, "minigame", "WEEK_OF_YEAR", "MINIGAME-WEEK")
create_combined_segment(df_max_71, "minigame", "SEGMENT_AGE", "MINIGAME-SEGMENT_AGE")
create_combined_segment(df_max_71, "hour", "SEGMENT_AGE", "HOUR-SEGMENT_AGE")
create_combined_segment(df_max_71, "used_premium_feature", "SEGMENT_AGE", "PREMIUM_FEATURE-SEGMENT_AGE")
create_combined_segment(df_max_71, "hour", "SEGMENT_SCREEN_NUM", "HOUR-SEGMENT_SCREEN_NUM")
create_combined_segment(df_max_71, "minigame", "hour", "MINIGAME_HOUR")



### Label Encoding > Binary Encoding & One-Hot Encoding
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

le = LabelEncoder()
enrolled_and_user = ["enrolled", "user"]
user_df = df_max_71[enrolled_and_user]
encoded_df = df_max_71.drop(enrolled_and_user + date_cols, axis=1)
encoded_df = encoded_df.drop("difference", axis=1)
cat_cols_encode, num_cols_encode, cat_but_car_encode = grab_col_names(encoded_df, cat_th=35, car_th=24)

binary_cols = [col for col in encoded_df.columns if encoded_df[col].dtype != "O" and encoded_df[col].nunique() == 2]



def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    label_encoder(encoded_df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


encoded_df2 = encoded_df.copy()
ohe_cols = [col for col in encoded_df2.columns if encoded_df2[col].nunique() > 2]
encoded_df2 = one_hot_encoder(encoded_df2, ohe_cols)


## Base Models ##
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc

y = user_df["enrolled"]
X = encoded_df2


def train_and_evaluate_models(X, y, models, plot=False):
    for model_name, model_class in models:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        model = model_class.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # For ROC-AUC

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        confusion_mat = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(confusion_mat, index=(0, 1), columns=(0, 1))

        # Plot the confusion matrix
        if plot:
            plt.figure(figsize=(10, 7))
            sns.set(font_scale=1.4)
            sns.heatmap(df_cm, annot=True, fmt='g')
            plt.title(f"{model_name} - Test Data Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
            plt.show()

        # Print the accuracy
        print(f"{model_name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}\n")


# Define the models
models = [("GradientBoostingClassifier", GradientBoostingClassifier(random_state=10)),
          ("RandomForestClassifier", RandomForestClassifier(random_state=10)),
          ("AdaBoostClassifier", AdaBoostClassifier(random_state=10)),
          ("LogisticRegression", LogisticRegression(max_iter=1000)),
          ("KNeighborsClassifier", KNeighborsClassifier()),
          ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=10)),
          ("XGBClassifier", XGBClassifier(eval_metric='logloss')),
          ("LGBMClassifier", LGBMClassifier(random_state=10))
          ]

train_and_evaluate_models(X, y, models)
# GradientBoostingClassifier - Accuracy: 0.7708, ROC-AUC: 0.8533
# RandomForestClassifier - Accuracy: 0.7925, ROC-AUC: 0.8718
# AdaBoostClassifier - Accuracy: 0.7614, ROC-AUC: 0.8441
# LogisticRegression - Accuracy: 0.7729, ROC-AUC: 0.8532
# KNeighborsClassifier - Accuracy: 0.6492, ROC-AUC: 0.6537
# DecisionTreeClassifier - Accuracy: 0.7376, ROC-AUC: 0.7214
# XGBClassifier - Accuracy: 0.7883, ROC-AUC: 0.8713
# LGBMClassifier - Accuracy: 0.7981, ROC-AUC: 0.8807

# LGBMClassifier has the best model performance. so I'm going to use that model. Now let's find the best parameters for
# cross-validations, then optimize the model.



### Hyperparameter Optimization
# LGBMClassifier
param_grid_lgbm = {'n_estimators': [155, 160, 165],
                   'learning_rate': [0.08, 0.09, 0.1],
                   'max_depth': [-11, -10, -9, -8]}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lgbm_model = LGBMClassifier(random_state=10).fit(X_train_scaled, y_train)


grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid_lgbm, scoring='roc_auc', cv=5)
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)
# Best Parameters: {'learning_rate': 0.09, 'max_depth': -11, 'n_estimators': 160}
print("Best ROC-AUC Score:", grid_search.best_score_)
# Best ROC-AUC Score: 0.8782453077223746


y_pred = grid_search.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy Score:", accuracy)
print("Test ROC-AUC Score:", roc_auc)
# Test Accuracy Score: 0.8013636819412413
# Test ROC-AUC Score: 0.7883364196896693


confusion_mat = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mat, index=(0, 1), columns=(0, 1))
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title(f"{lgbm_model} - Test Data Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
plt.show()


import shap


def plot_summary(model, features, feature_names=None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # If feature_names is not provided, use default indexing
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(features.shape[1])]

    # Create a SHAP summary plot
    shap.summary_plot(shap_values[1], features, feature_names=feature_names, show=False)
    plt.show()


plot_summary(lgbm_model, X_train_scaled, feature_names=X_train.columns)



final_results = pd.concat([y_test, user_df["user"]], axis=1).dropna()
final_results["Predicted"] = y_pred
final_results["Probability"] = y_pred_proba
final_results = final_results[["user", "enrolled", "Predicted", "Probability"]].reset_index(drop=True)
final_results_sorted = final_results.sort_values(by='Probability', ascending=False).reset_index(drop=True)



# Calculate SHAP values using the trained model
explainer = shap.TreeExplainer(lgbm_model)  # If using LGBMClassifier
shap_values = explainer.shap_values(X_test_scaled)

# Get SHAP values for the specific user
user_index = final_results[final_results['user'] == 176810].index[0]
user_shap_values = shap_values[0][user_index, :]  # Access the first element if shap_values is a list

# Match SHAP values with features
shap_df = pd.DataFrame(list(zip(X.columns, user_shap_values)), columns=['Feature', 'SHAP Value'])

# Sort SHAP values by magnitudes
shap_df = shap_df.sort_values(by='SHAP Value', ascending=False)

# Visualize SHAP values
shap.summary_plot(shap_values[0], X_test_scaled, feature_names=X.columns)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Eşik değeri seçimi için ROC eğrisini görselleştirin
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# tests:
# Best Parameters: {'learning_rate': 0.09, 'max_depth': -5, 'n_estimators': 155}
# Best ROC-AUC Score: 0.8781587627471866
# Test Accuracy Score: 0.8007620575553995
# Test ROC-AUC Score: 0.7878010588642208
# Best Parameters: {'learning_rate': 0.09, 'max_depth': -7, 'n_estimators': 160}
# Best ROC-AUC Score: 0.8782453077223746
# Test Accuracy Score: 0.8013636819412413
# Test ROC-AUC Score: 0.7883364196896693
# Best Parameters: {'learning_rate': 0.09, 'max_depth': -9, 'n_estimators': 160}
# Best ROC-AUC Score: 0.8782453077223746
# Test Accuracy Score: 0.8013636819412413
# Test ROC-AUC Score: 0.7883364196896693
# Best Parameters: {'learning_rate': 0.09, 'max_depth': -11, 'n_estimators': 160}
# Best ROC-AUC Score: 0.8782453077223746
# Test Accuracy Score: 0.8013636819412413
# Test ROC-AUC Score: 0.7883364196896693