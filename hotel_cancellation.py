# -*- coding: utf-8 -*-
"""
Hotel Reservation Cancellation Prediction Using Machine Learning
Course: Predictive Analytics
Author: Manjari N.K.

Run this file from the folder that contains:
    data/Hotel Reservations.csv
    figures/            (an empty folder; charts are saved here)
"""

# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# =====================================================
# 2. LOAD AND INSPECT THE DATASET
# =====================================================
df = pd.read_csv("Hotel Reservations.csv")

print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nInfo:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("Unique Booking IDs:", df['Booking_ID'].nunique())

print("\nBooking status counts:")
print(df['booking_status'].value_counts())
print("\nBooking status percentage:")
print(df['booking_status'].value_counts(normalize=True) * 100)

# =====================================================
# 3. INVESTIGATE UNUSUAL VALUES (do not delete blindly)
# =====================================================
# 3a. Invalid calendar dates (for example 29 February in a non-leap year)
dates = pd.to_datetime(dict(year=df['arrival_year'],
                            month=df['arrival_month'],
                            day=df['arrival_date']), errors='coerce')
print("\nInvalid arrival dates:", dates.isna().sum())
print(df[dates.isna()][['arrival_year', 'arrival_month', 'arrival_date']].drop_duplicates())

# 3b. Zero average room price
zero_price = df[df['avg_price_per_room'] == 0]
print("\nRows with zero room price:", len(zero_price))
print(zero_price['market_segment_type'].value_counts())
print(zero_price['booking_status'].value_counts())

# 3c. Zero adults
zero_adults = df[df['no_of_adults'] == 0]
print("\nRows with zero adults:", len(zero_adults))
print(zero_adults['no_of_children'].describe())

# 3d. Bookings with zero nights
print("\nRows with zero total nights:",
      ((df['no_of_weekend_nights'] + df['no_of_week_nights']) == 0).sum())

# =====================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================
# Numeric copy of the target for calculating cancellation rates (1 = Canceled)
df['cancelled'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})

# ---- 4.1 Cancellation distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='booking_status', data=df)
plt.title("Distribution of Booking Status")
plt.xlabel("Booking Status")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.show()

# ---- 4.2 Market segment (our customer-group analysis)
plt.figure(figsize=(8, 4))
sns.countplot(x='market_segment_type', hue='booking_status', data=df)
plt.title("Bookings by Market Segment and Status")
plt.xlabel("Market Segment")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.show()

segment_rate = df.groupby('market_segment_type')['cancelled'].mean() * 100
segment_rate = segment_rate.reset_index()
segment_rate.columns = ['market_segment_type', 'cancellation_rate']
print("\nCancellation rate (%) by market segment:")
print(segment_rate)

plt.figure(figsize=(8, 4))
sns.barplot(x='market_segment_type', y='cancellation_rate', data=segment_rate)
plt.title("Cancellation Rate (%) by Market Segment")
plt.xlabel("Market Segment")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.show()

# ---- 4.3 Repeated guest vs new guest
repeat_rate = df.groupby('repeated_guest')['cancelled'].mean() * 100
repeat_rate = repeat_rate.reset_index()
repeat_rate.columns = ['repeated_guest', 'cancellation_rate']
print("\nCancellation rate (%) by repeated guest (0 = new, 1 = repeated):")
print(repeat_rate)

plt.figure(figsize=(5, 4))
sns.barplot(x='repeated_guest', y='cancellation_rate', data=repeat_rate)
plt.title("Cancellation Rate (%) - New vs Repeated Guests")
plt.xlabel("Repeated Guest (0 = No, 1 = Yes)")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.show()

# ---- 4.4 Lead time
print("\nAverage lead time by booking status:")
print(df.groupby('booking_status')['lead_time'].describe())

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='lead_time', hue='booking_status', bins=40)
plt.title("Lead Time Distribution by Booking Status")
plt.xlabel("Lead Time (days)")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='booking_status', y='lead_time', data=df)
plt.title("Lead Time vs Booking Status")
plt.xlabel("Booking Status")
plt.ylabel("Lead Time (days)")
plt.tight_layout()
plt.show()

# ---- 4.5 Room type and meal plan
room_rate = df.groupby('room_type_reserved')['cancelled'].mean() * 100
room_rate = room_rate.reset_index()
room_rate.columns = ['room_type_reserved', 'cancellation_rate']
meal_rate = df.groupby('type_of_meal_plan')['cancelled'].mean() * 100
meal_rate = meal_rate.reset_index()
meal_rate.columns = ['type_of_meal_plan', 'cancellation_rate']
print("\nCancellation rate (%) by room type:")
print(room_rate)
print("\nCancellation rate (%) by meal plan:")
print(meal_rate)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.barplot(x='room_type_reserved', y='cancellation_rate', data=room_rate)
plt.title("Cancellation Rate (%) by Room Type")
plt.xlabel("Room Type")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.barplot(x='type_of_meal_plan', y='cancellation_rate', data=meal_rate)
plt.title("Cancellation Rate (%) by Meal Plan")
plt.xlabel("Meal Plan")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---- 4.6 Price
print("\nAverage room price by booking status:")
print(df.groupby('booking_status')['avg_price_per_room'].describe())

plt.figure(figsize=(6, 4))
sns.boxplot(x='booking_status', y='avg_price_per_room', data=df)
plt.title("Average Room Price vs Booking Status")
plt.xlabel("Booking Status")
plt.ylabel("Average Price per Room")
plt.tight_layout()
plt.show()

# ---- 4.7 Special requests
request_rate = df.groupby('no_of_special_requests')['cancelled'].mean() * 100
request_rate = request_rate.reset_index()
request_rate.columns = ['no_of_special_requests', 'cancellation_rate']
print("\nCancellation rate (%) by number of special requests:")
print(request_rate)

plt.figure(figsize=(6, 4))
sns.barplot(x='no_of_special_requests', y='cancellation_rate', data=request_rate)
plt.title("Cancellation Rate (%) by Number of Special Requests")
plt.xlabel("Number of Special Requests")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.show()

# ---- 4.8 Arrival month and arrival year
month_rate = df.groupby('arrival_month')['cancelled'].mean() * 100
month_rate = month_rate.reset_index()
month_rate.columns = ['arrival_month', 'cancellation_rate']
year_rate = df.groupby('arrival_year')['cancelled'].mean() * 100
year_rate = year_rate.reset_index()
year_rate.columns = ['arrival_year', 'cancellation_rate']
print("\nCancellation rate (%) by arrival month:")
print(month_rate)
print("\nCancellation rate (%) by arrival year:")
print(year_rate)

plt.figure(figsize=(8, 4))
sns.barplot(x='arrival_month', y='cancellation_rate', data=month_rate)
plt.title("Cancellation Rate (%) by Arrival Month")
plt.xlabel("Arrival Month")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.show()

# ---- 4.9 Correlation heatmap (numerical columns only, before encoding)
corr = df.corr(numeric_only=True)
plt.figure(figsize=(13, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap of Numerical Features (cancelled: 1 = Canceled)")
plt.tight_layout()
plt.show()

print("\nCorrelation of every numerical feature with 'cancelled':")
print(corr['cancelled'].sort_values(ascending=False))

# =====================================================
# 5. PREPROCESSING
# =====================================================
df_model = df.copy()

# Drop: ID (identifier), booking_status (the target in text form - leakage if kept),
# arrival_year (only two years of data) and arrival_date (day of month, weak, holds the invalid dates)
df_model.drop(columns=['Booking_ID', 'booking_status', 'arrival_year', 'arrival_date'],
              inplace=True)

# Encode the text columns with LabelEncoder
cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
label_enc = LabelEncoder()
for col in cat_cols:
    df_model[col] = label_enc.fit_transform(df_model[col])
    print("\nEncoding for", col, ":")
    print(dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_))))

print("\nColumns used in the model:")
print(df_model.columns)
print(df_model.info())

# =====================================================
# 6. SELECT X AND y, TRAIN-TEST SPLIT, SCALING
# =====================================================
X = df_model.drop('cancelled', axis=1)
y = df_model['cancelled']
print("\nFeatures used:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set:", X_train.shape, " Test set:", X_test.shape)
print("Cancelled share in train:", round(y_train.mean() * 100, 2), "%")
print("Cancelled share in test :", round(y_test.mean() * 100, 2), "%")

# Scaling: fit on the training data only, then transform the test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 7. TRAIN MODELS AND PREDICT
# (scaled data for Logistic Regression, KNN, SVM; original data for the rest)
# =====================================================
# 7.1 Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# 7.2 K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# 7.3 Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# 7.4 Support Vector Machine (this one takes a minute or two)
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# 7.5 Decision Tree
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# 7.6 Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# =====================================================
# 8. EVALUATION
# =====================================================
predictions = {
    "Logistic Regression": y_pred_lr,
    "KNN": y_pred_knn,
    "Gaussian Naive Bayes": y_pred_nb,
    "SVM": y_pred_svm,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
}

results = []
cms = {}
for name in predictions:
    y_pred = predictions[name]
    cm = confusion_matrix(y_test, y_pred)
    cms[name] = cm
    acc = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print(name)
    print("==============================")
    print("Accuracy:", round(acc, 4))
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Not Canceled', 'Canceled']))

    # Metrics for the Canceled class (class 1), worked out from the confusion matrix
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    results.append([name, acc, precision, recall, f1, tn, fp, fn, tp])

comparison = pd.DataFrame(results,
                          columns=['Model', 'Accuracy', 'Precision (Canceled)',
                                   'Recall (Canceled)', 'F1 (Canceled)',
                                   'TN', 'FP', 'FN', 'TP'])
print("\n\nMODEL COMPARISON TABLE")
print(comparison.round(4).to_string(index=False))
comparison.round(4).to_csv("model_comparison.csv", index=False)

# ---- Confusion matrix heatmaps for all six models
plt.figure(figsize=(15, 8))
i = 1
for name in cms:
    plt.subplot(2, 3, i)
    sns.heatmap(cms[name], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Canceled', 'Canceled'],
                yticklabels=['Not Canceled', 'Canceled'])
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    i = i + 1
plt.tight_layout()
plt.show()

# ---- Bar chart comparing the models
comparison.set_index('Model')[['Accuracy', 'Precision (Canceled)',
                               'Recall (Canceled)', 'F1 (Canceled)']].plot(
    kind='bar', figsize=(11, 5))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.legend(loc='lower right')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# =====================================================
# 9. INTERPRETATION OF TREE-BASED MODELS
# =====================================================
# 9.1 First levels of the Decision Tree (real, unscaled values)
plt.figure(figsize=(22, 10))
plot_tree(dt,
          feature_names=X.columns,
          class_names=["Not Canceled", "Canceled"],
          filled=True,
          rounded=True,
          max_depth=3,
          fontsize=9)
plt.title("Decision Tree - First 3 Levels")
plt.tight_layout()
plt.show()
print("\nDecision Tree depth:", dt.get_depth(), " leaves:", dt.get_n_leaves())

# 9.2 Random Forest feature importance
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRandom Forest feature importance:")
print(importance.round(4))

plt.figure(figsize=(8, 5))
sns.barplot(x=importance.values, y=importance.index)
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# =====================================================
# 10. LEAKAGE CHECK: remove 'no_of_special_requests' and re-run two models
# (guests can add special requests after booking, so this feature is borderline)
# =====================================================
X2 = X.drop(columns=['no_of_special_requests'])
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42, stratify=y)

rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf2.fit(X2_train, y2_train)
y_pred_rf2 = rf2.predict(X2_test)

sc2 = StandardScaler()
X2_train_scaled = sc2.fit_transform(X2_train)
X2_test_scaled = sc2.transform(X2_test)
lr2 = LogisticRegression(max_iter=1000)
lr2.fit(X2_train_scaled, y2_train)
y_pred_lr2 = lr2.predict(X2_test_scaled)

check = {"Random Forest": (y_pred_rf, y_pred_rf2),
         "Logistic Regression": (y_pred_lr, y_pred_lr2)}
check_rows = []
for name in check:
    for label, pred in zip(["with special requests", "without special requests"], check[name]):
        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        check_rows.append([name, label, accuracy_score(y_test, pred), p, r,
                           2 * p * r / (p + r)])
check_df = pd.DataFrame(check_rows, columns=['Model', 'Feature set', 'Accuracy',
                                             'Precision (Canceled)', 'Recall (Canceled)',
                                             'F1 (Canceled)'])
print("\nLEAKAGE / SENSITIVITY CHECK")
print(check_df.round(4).to_string(index=False))
check_df.round(4).to_csv("sensitivity_check.csv", index=False)
