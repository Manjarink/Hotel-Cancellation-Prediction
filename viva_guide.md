# Viva Guide: Hotel Reservation Cancellation Prediction

Read Part A until you can explain each block in your own words. Part B has the questions an examiner is likely to ask. All numbers come from the real run.

---

## Part A. How your code works (block by block)

**1. Imports.** `pandas` loads and handles tables, `numpy` is for numbers, `matplotlib` and `seaborn` draw charts, `train_test_split` divides the data, `LabelEncoder` turns text into numbers, `StandardScaler` puts features on one scale, then the six models and three metric functions.

**2. Load and inspect.** `pd.read_csv` reads the file. `head()`/`tail()` show first and last rows, `shape` gives (36275, 19), `info()` shows types and non-null counts, `describe()` gives mean, min, max. `isnull().sum()` counts missing values, `duplicated().sum()` counts repeated rows.

**3. Unusual values.** `pd.to_datetime(..., errors='coerce')` builds a date from year, month and day and gives NaT for impossible dates (29 Feb 2018). `isna().sum()` counts them: 37. Then we filter rows with `df[df['avg_price_per_room'] == 0]` and look at their segment and status.

**4. The `cancelled` column.** `.map({'Not_Canceled': 0, 'Canceled': 1})` makes a numeric target. Mean of this column is the cancellation rate.

**5. EDA charts.**
- `sns.countplot` counts rows per category (with `hue` it splits by status)
- `df.groupby('x')['cancelled'].mean() * 100` is the cancellation rate (%) per group
- `sns.barplot` draws those rates
- `sns.histplot` shows a distribution, `sns.boxplot` shows median, spread and outliers
- `df.corr(numeric_only=True)` + `sns.heatmap(annot=True)` shows correlations
- `plt.savefig` saves the figure, `plt.show` displays it

**6. Preprocessing.** `drop(columns=[...])` removes the ID, the text target, `arrival_year` and `arrival_date`. The loop `for col in cat_cols` applies `LabelEncoder().fit_transform` to each text column. Printing `dict(zip(label_enc.classes_, ...))` shows which number each category got.

**7. X, y and split.** `X` = all feature columns, `y` = `cancelled`. `train_test_split(test_size=0.2, random_state=42, stratify=y)` keeps 80% for training, 20% for testing, always the same split, and the same 32.76% cancelled share in both parts.

**8. Scaling.** `fit_transform` on training data learns the mean and spread and applies them. `transform` on test data reuses those numbers. Fitting on the test data would leak information.

**9. Models.** Each has the same three steps: create, `fit(X_train, y_train)`, `predict(X_test)`. Scaled data goes to Logistic Regression, KNN and SVM (they depend on feature scale). Trees and Naive Bayes use the original data.

**10. Evaluation loop.** For each model: `confusion_matrix`, `accuracy_score`, `classification_report`. `cm.ravel()` unpacks the matrix into TN, FP, FN, TP, from which we compute precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R) for the Canceled class.

**11. Interpretation.** `plot_tree(dt, max_depth=3, ...)` draws the first levels of the tree. `rf.feature_importances_` gives each feature's importance, shown as a bar chart.

**12. Sensitivity check.** Repeats Random Forest and Logistic Regression without `no_of_special_requests`.

---

## Part B. Questions and answers

### Basic project questions
**Q1. What is your project about?**
It predicts whether a hotel booking will be cancelled, using information known when the booking is made, and compares six classification algorithms.

**Q2. Why is this a classification problem?**
The answer is one of two classes, Canceled or Not_Canceled, not a number.

**Q3. Why is it useful?**
A hotel can plan overbooking, send reminders or ask for deposits for bookings that look risky.

**Q4. What is the problem statement?**
Can we predict whether a hotel reservation will be cancelled or not cancelled using booking information available at the time the reservation is made?

**Q5. Which model did you choose and why?**
Random Forest. It had the best accuracy (0.9012), precision for cancellations (0.8843) and F1 (0.8420), with recall 0.8035 (the highest among the useful models). I did not choose by accuracy alone.

### Dataset questions
**Q6. Where is the data from, and how big is it?**
Kaggle (ahsan81). 36,275 rows and 19 columns.

**Q7. What is the target?**
`booking_status`. 67.24% Not_Canceled, 32.76% Canceled.

**Q8. Is the data balanced?**
Moderately imbalanced. Always predicting "Not Canceled" would give about 67% accuracy, which is why accuracy alone is not enough.

**Q9. Were there missing values or duplicates?**
None. I checked with `isnull().sum()` and `duplicated().sum()`.

**Q10. What odd values did you find?**
37 invalid dates (all 29 Feb 2018), 545 zero-price rows (354 Complementary, 191 Online, 539 not cancelled), 139 zero-adult rows (all with children), 78 rows with zero nights.

**Q11. Did you remove them?**
No. I investigated first. Zero prices are mostly complimentary stays, zero-adult bookings all have children, and the invalid dates sit in a column I don't use.

**Q12. Is there a "customer type" column?**
No. I used market segment and repeated-guest status to describe customer groups.

### Preprocessing questions
**Q13. Which columns did you drop and why?**
`Booking_ID` (an identifier), `booking_status` (it is the target, keeping it would be leakage), `arrival_year` (only two values, the model would learn the year), `arrival_date` (weak and holds the invalid dates).

**Q14. What is data leakage?**
Using information in training that would not be available when predicting, or that reveals the answer. Example: keeping the target column as a feature.

**Q15. Which feature is borderline for leakage?**
`no_of_special_requests`, because guests can add requests after booking. I tested the model without it: Random Forest F1 fell from 0.8420 to 0.7484.

**Q16. What does LabelEncoder do, and what is its weakness?**
It turns each category into a number (Meal Plan 1 = 0, and so on). The weakness is that it creates an artificial order (Online = 4 is "more than" Offline = 3), which can mislead Logistic Regression, KNN and SVM.

**Q17. Why scale the data?**
KNN and SVM use distances and Logistic Regression uses weights, so a feature with large values (like price) would dominate. Trees don't need scaling.

**Q18. Why fit the scaler only on training data?**
So the test data stays unseen. Fitting on all data would leak test information.

**Q19. What does `random_state=42` do?**
It makes the split the same every run, so results can be repeated.

**Q20. What does `stratify=y` do?**
Keeps the cancelled share (32.76%) the same in training and test sets.

### EDA questions
**Q21. What affects cancellation most?**
Lead time. Cancelled bookings have a mean lead time of 139 days versus 59 days for completed ones (correlation +0.44).

**Q22. Which segment cancels most and least?**
Online 36.51% (most), Corporate 10.91%, Complementary 0.00% (least).

**Q23. What did special requests show?**
0 requests: 43.21% cancelled. 1 request: 23.77%. 2 requests: 14.60%. 3 or more: 0%.

**Q24. Why is the month chart tricky?**
Data cover Jul 2017 to Dec 2018, and 2017 had a much lower cancellation rate (14.75% vs 36.71%). Month and year effects are mixed.

**Q25. What is a correlation heatmap and what are its limits?**
It shows how strongly pairs of numeric columns move together (-1 to +1). It shows linear association only, and association is not cause.

### Algorithm questions
**Q26. Logistic Regression?**
A linear model that turns a weighted sum of features into a probability between 0 and 1. If the probability passes 0.5 it predicts the positive class. Result: accuracy 0.8094, recall 0.6226, so it misses many cancellations.

**Q27. KNN?**
Finds the 5 nearest training bookings (Euclidean distance) and takes a majority vote. Needs scaling. Accuracy 0.8579.

**Q28. Gaussian Naive Bayes?**
Uses Bayes' theorem and assumes features are independent and normally distributed within each class. It performed badly (accuracy 0.3927) because many features are skewed. It flagged about 93% of bookings as cancellations.

**Q29. SVM?**
Finds the boundary that separates the classes with the widest margin. I used the RBF kernel, which allows curved boundaries. Accuracy 0.8459, precision 0.8233 but recall only 0.6744.

**Q30. Decision Tree?**
A flowchart of yes/no questions on features. First question: lead_time ≤ 151.5. It reached accuracy 0.8702, but it grew to depth 40, so it may over-fit.

**Q31. Random Forest?**
100 trees, each trained on a random sample of rows and features, whose votes are combined. This reduces the over-fitting of one tree. Accuracy 0.9012.

**Q32. Why did Random Forest beat the others?**
It handles mixed features, non-linear patterns and interactions (for example, lead time together with price) without scaling, and it averages away the errors of single trees.

### Confusion matrix and metrics
**Q33. Explain the confusion matrix for Random Forest.** `[[4628, 250], [467, 1910]]`
4,628 completed bookings correctly predicted (TN), 250 completed bookings wrongly flagged as cancellations (FP), 467 cancellations missed (FN), 1,910 cancellations caught (TP).

**Q34. Define accuracy, precision, recall and F1.**
Accuracy = share of correct predictions. Precision = of those predicted cancelled, how many truly were (1910/2160 = 0.8843). Recall = of all real cancellations, how many were caught (1910/2377 = 0.8035). F1 = balance of the two.

**Q35. Why not just use accuracy?**
Because of class imbalance and unequal error costs. Naive Bayes shows it: recall 0.9882 but accuracy 0.3927.

**Q36. Which is worse, a false positive or a false negative?**
It depends on the hotel. A false negative means an empty room that was not planned for. A false positive means a reliable guest was treated as risky. The hotel should choose its threshold based on these costs.

**Q37. Why choose by F1 for the Canceled class?**
It needs both good precision and good recall on the class we care about. A model can have high accuracy and still be poor at finding cancellations.

### Limitations and improvements
**Q38. What are the limitations?**
One data source; random split (not time-based) and mixed month/year effects; LabelEncoder's artificial order; no tuning or cross-validation; special requests may not be known at booking; deep single tree; results show association, not cause.

**Q39. How would you improve it?**
Cross-validation, hyperparameter tuning, one-hot encoding, a time-based split, a cost-based threshold, more features (deposit, payment type, local events).

**Q40. Can the hotel use this model today?**
Not directly. First confirm that special requests are known at prediction time, test on newer bookings and check costs of errors.

### Likely "trap" questions
- *"Your accuracy is 90%, is it good?"* Compare with the 67% baseline, then give recall and precision for cancellations.
- *"Feature importance says lead time is key. Does shortening lead time reduce cancellations?"* No. Importance shows what the model uses, not what causes cancellation.
- *"Why did you not tune the models?"* The course practicals use standard settings and I wanted a fair comparison at the same settings. Tuning is listed in future scope.
- *"Did you check your numbers?"* Yes, every number in the report is printed by the script (see `run_output.txt`), and you can re-run it.
