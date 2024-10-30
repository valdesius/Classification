import pandas as pd
df = pd.read_csv('sample_data/data.csv')
df.head()

df.drop('parent_age', axis=1, inplace=True)
df.drop('house_area', axis=1, inplace=True)

df.describe()

print(df.nunique())

missing_info = df.isnull().sum()
missing_percent = (missing_info / df.shape[0]) * 100
missing_data = pd.DataFrame({'Missing Values': missing_info, 'Percentage': missing_percent})
print(missing_data)

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x=df['average_grades'])
plt.title('Средние оценки')
plt.show()

sns.histplot(df['average_grades'], kde=True)
plt.title('Средние оценки')

def handle_outliers(data, column):
  q1 = data[column].quantile(0.25)
  q3 = data[column].quantile(0.75)
  iqr = q3 - q1
  lower_whisker = q1 - 1.5 * iqr
  upper_whisker = q3 + 1.5 * iqr

  data[column] = data[column].apply(lambda x: lower_whisker if x < lower_whisker else (upper_whisker if x  > upper_whisker else x))

data = handle_outliers(df, 'average_grades')
sns.boxplot(x=df['average_grades'])
plt.title('Средние оценки')
plt.show()

sns.histplot(df['average_grades'], kde=True)
plt.title('Средние оценки')

duplicate_rows = df[df.duplicated()]
print(duplicate_rows)

sns.countplot(x="residence", hue="will_go_to_college", data=df, palette="Set1")
plt.xlabel('Место жительства')
plt.ylabel('Count')
plt.title('Распределение желающих поступить в колледж по месту жительства')
plt.legend(title='Пойдут в колледж')
plt.show()

from sklearn.preprocessing import MinMaxScaler
quantitative_cols = ['parent_salary', 'average_grades']
scaler = MinMaxScaler()
df[quantitative_cols] = scaler.fit_transform(df[quantitative_cols])

print(df.head())
sns.pairplot(df, vars=quantitative_cols, kind="reg")

quantitative_vars = ['parent_salary', 'average_grades']
df_quantitative = df[quantitative_vars]

plt.figure(figsize=(10, 8))
correlation_matrix = df_quantitative.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Корреляционная матрица')
plt.show()

from scipy.stats import chi2_contingency
categorical_vars = ['type_school', 'school_accreditation', 'gender', 'interest', 'residence', 'parent_was_in_college', 'will_go_to_college']

for var in categorical_vars:
    if var not in df.columns:
        print(f"Column '{var}' is not present in the DataFrame.")

def chi_square_test(var1, var2):
    cross_tab = pd.crosstab(df[var1], df[var2])
    chi2, p, _, _ = chi2_contingency(cross_tab)
    return chi2, p
results = {}

for i in range(len(categorical_vars)):
    for j in range(i + 1, len(categorical_vars)):
        var1 = categorical_vars[i]
        var2 = categorical_vars[j]
        if var1 in df.columns and var2 in df.columns:
            chi2, p = chi_square_test(var1, var2)
            results[(var1, var2)] = {'Chi-square value': chi2, 'P-value': p}

for pair, result in results.items():
    print(f"Pair: {pair}")
    print(f"Chi-square value: {result['Chi-square value']}")
    print(f"P-value: {result['P-value']}")
    print()

from sklearn.preprocessing import LabelEncoder

categorical_columns = df.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()

for col in categorical_columns:
    df[col + '_LabelEncoded'] = label_encoder.fit_transform(df[col])
df.head()

X = df.drop(columns=['will_go_to_college', 'type_school', 'school_accreditation', 'gender', 'interest', 'residence'])
y = df['will_go_to_college']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mutual_info = mutual_info_regression(x_train, y_train, random_state=42)

feature_names = x_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': mutual_info})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Взаимная информация')
plt.title('Важность признаков')
plt.show()

print(feature_importance_df)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)

feature_importances = model.feature_importances_
feature_names = x_train.columns

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Важность')
plt.title('Важность признаков в модели случайный лес')
plt.show()

print(feature_importance_df)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

df['will_go_to_college'] = df['will_go_to_college'].astype(int)
selected_cols = ['parent_salary', 'average_grades', 'parent_was_in_college', 'type_school_LabelEncoded', 'school_accreditation_LabelEncoded', 'gender_LabelEncoded', 'interest_LabelEncoded', 'residence_LabelEncoded']
X = df[selected_cols]
y = df['will_go_to_college']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVC': SVC(),
    'KNeighbors': KNeighborsClassifier(),
    'MLP': MLPClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))

from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Report for {name}: {report}")

    report['accuracy'] = accuracy
    results[name] = report
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))

print(f"Results: {results}")

best_model = max(results, key=lambda x: results[x]['accuracy'])
print(f"\nBest Model: {best_model}")
print(f"Accuracy: {results[best_model]['accuracy']}")
print(f"Macro Avg F1-score: {results[best_model]['macro avg']['f1-score']}")
print(f"Weighted Avg F1-score: {results[best_model]['weighted avg']['f1-score']}")

from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)

y_pred_reg = model_reg.predict(X_test)  # Предсказание на тестовом наборе

mse = mean_squared_error(y_test, y_pred_reg)  # Среднеквадратичная ошибка (MSE)
print(f'Mean Squared Error: {mse}')

mae = mean_absolute_error(y_test, y_pred_reg)  # Средняя абсолютная ошибка (MAE)
print(f'Mean Absolute Error: {mae}')

r2 = r2_score(y_test, y_pred_reg)  # Коэффициент детерминации (R²)
print(f'R-squared: {r2}')

