import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings

# Load the heart dataset from a CSV file
heart_df = pd.read_csv("heart1.csv")

# Split the dataset into features (X) and target labels (y)
X = heart_df.drop('a1p2', axis=1)
y = heart_df['a1p2']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features (scaling)
sc = StandardScaler()                 # Create the standard scaler
sc.fit(X_train)                       # Compute the required transformation
X_train_std = sc.transform(X_train)   # Apply the transformation to the training data
X_test_std = sc.transform(X_test)     # Apply the same transformation to the test data


warnings.filterwarnings('ignore')
## Perceptron analysis

print("Perceptron analysis: \n")

ppn = Perceptron(max_iter=6, tol=1e-2, eta0=0.001,
                 fit_intercept=True, random_state=0, verbose=False)
ppn.fit(X_train_std, y_train)

y_pred_ppn = ppn.predict(X_test_std)

# print("In test: \n")
# print('Misclassified samples: %d' % (y_test != y_pred_ppn).sum())
print('%.2f' % accuracy_score(y_test, y_pred_ppn))

# # Combining data and running analysis
X_combined_std_ppn = np.vstack((X_train_std, X_test_std))
y_combined_ppn = np.hstack((y_train, y_test))
y_combined_pred_ppn = ppn.predict(X_combined_std_ppn)

# print('Misclassified combined samples: %d' % (y_combined_ppn != y_combined_pred_ppn).sum())
# print('Combined Accuracy: %.2f' % accuracy_score(y_combined_ppn, y_combined_pred_ppn), "\n")

## Logistic Regression
print("Logistic Regression analysis: \n")

C_VAL_LR = 1  # Chosen after testing for values 1, 5, 10, 50, 100
lr = LogisticRegression(C=C_VAL_LR, solver='liblinear', \
                        multi_class='ovr', random_state=0)
lr.fit(X_train_std, y_train)

y_pred_lr = lr.predict(X_test_std)

# print("In test: \n")
# print('Misclassified samples: %d' % (y_test != y_pred_lr).sum())
print('%.2f' % accuracy_score(y_test, y_pred_lr))

# Combining data and running analysis
X_combined_std_lr = np.vstack((X_train_std, X_test_std))
y_combined_lr = np.hstack((y_train, y_test))
y_combined_pred_lr = lr.predict(X_combined_std_lr)

# print('Misclassified combined samples: %d' % (y_combined_lr != y_combined_pred_lr).sum())
# print('Combined Accuracy: %.2f' % accuracy_score(y_combined_lr, y_combined_pred_lr), "\n")


## Support Vector Machine
print("Support Vector Machine analysis: \n")

C_VAL_SVM = 0.1  # Chosen after testing for values 0.01, 0.1, 0.5, 1.0, 10.0

svm = SVC(kernel='linear', C=C_VAL_SVM, random_state=0)
svm.fit(X_train_std, y_train)

y_pred_svm = svm.predict(X_test_std)

# print("In test: \n")
# print('Misclassified samples: %d' % (y_test != y_pred_svm).sum())
print('%.2f' % accuracy_score(y_test, y_pred_svm))

# Combining data and running analysis
X_combined_std_svm = np.vstack((X_train_std, X_test_std))
y_combined_svm = np.hstack((y_train, y_test))
y_combined_pred_svm = svm.predict(X_combined_std_svm)

# print('Misclassified combined samples: %d' % (y_combined_svm != y_combined_pred_svm).sum())
# print('Combined Accuracy: %.2f' % accuracy_score(y_combined_svm, y_combined_pred_svm), "\n")


## Decision Tree Learning
print("Decision Tree Learning analysis: \n")

tree = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=0)
tree.fit(X_train
         , y_train)

y_pred_dtree = tree.predict(X_test)

# print("In test: \n")
# print('Misclassified samples: %d' % (y_test != y_pred_dtree).sum())
print('%.2f' % accuracy_score(y_test, y_pred_dtree))

# Combining data and running analysis
X_combined_dtree = np.vstack((X_train, X_test))
y_combined_dtree = np.hstack((y_train, y_test))
y_combined_pred_dtree = tree.predict(X_combined_dtree)

# print('Misclassified combined samples: %d' % \
#        (y_combined_dtree != y_combined_pred_dtree).sum())
# print('Combined Accuracy: %.2f' % accuracy_score(y_combined_dtree, y_combined_pred_dtree))
# input()

## Random Forest
print("Random Forest analysis: \n")

trees = 51  # Number of trees
print("Number of trees: ", trees)

forest = RandomForestClassifier(criterion='entropy', n_estimators=trees, \
                                random_state=1, n_jobs=4)
forest.fit(X_train, y_train)

y_pred_rf = forest.predict(X_test)

# print("In test: \n")
# print('Misclassified samples: %d' % (y_test != y_pred_rf).sum())
print('%.2f' % accuracy_score(y_test, y_pred_rf))

# Combining data and running analysis
X_combined_rf = np.vstack((X_train, X_test))
y_combined_rf = np.hstack((y_train, y_test))
y_combined_pred_rf = forest.predict(X_combined_rf)

# print('Misclassified combined samples: %d' % (y_combined_rf != y_combined_pred_rf).sum())
# print('Combined Accuracy: %.2f' % accuracy_score(y_combined_rf, y_combined_pred_rf), "\n")
# input()

## K-Nearest Neighbor
print("K-Nearest Neighbor analysis: \n")

neighs = 5  # Chosen after testing values 1, 2, 3, 4, 5, 51
print(neighs, 'neighbors')

knn = KNeighborsClassifier(n_neighbors=neighs, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

y_pred_knn = knn.predict(X_test_std)

# print("In test: \n")
# print('Misclassified samples: %d' % (y_test != y_pred_knn).sum())
print('%.2f' % accuracy_score(y_test, y_pred_knn))

# Combining data and running analysis
X_combined_std_knn = np.vstack((X_train_std, X_test_std))
y_combined_knn = np.hstack((y_train, y_test))
y_combined_pred_knn = knn.predict(X_combined_std_knn)

# print('Misclassified samples: %d' % (y_combined_knn != y_combined_pred_knn).sum())
# print('Combined Accuracy: %.2f' % \
       #  accuracy_score(y_combined_knn, y_combined_pred_knn), "\n")


classifiers = [ppn, lr, svm, tree, forest, knn]

def combine(classifiers, X):
    predictions = np.array([clf.predict(X) for clf in classifiers])
    sum_of_methods = np.sum(predictions, axis=0)
    return sum_of_methods

input()
# Combine top 3 methods
top_3_classifiers = [knn,lr,ppn ]
sum = combine(top_3_classifiers, X_test_std)
combined_results = np.where(sum > 4.5, 2, 1)

print(f"Ensemble with three methods: {accuracy_score(y_test, combined_results) * 100:.2f}%")
input()

# Adding the 4th best method
top_4_classifiers = [knn,lr, ppn, svm]
sum = combine(top_4_classifiers, X_test_std)
combined_results = np.where(sum >= 6, 2, 1)

print("Ties counted as 'Yes'")
print(f"Ensemble with four methods: {accuracy_score(y_test, combined_results) * 100:.2f}%")
input()

# Adding the 5th best method
top_5_classifiers = [knn,lr,ppn,svm,forest]
sum = combine(top_5_classifiers, X_test_std)
combined_results = np.where(sum > 7.5, 2, 1)

print(f"Ensemble with five methods: {accuracy_score(y_test, combined_results) * 100:.2f}%")
input()