import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def grid_search(x_train_flattened, y_train_labels, x_test_flattened, y_test_labels):
    param_grids = {
        'Logistic Regression': {
            'model': LogisticRegression(),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]  # Limiting the iterations
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [5, 10, 20],  # Reduced depth
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10, 50],  # Reduced estimators
                'max_depth': [5, 10],
                'min_samples_split': [2, 5]
            }
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 11],  # Reduced number of neighbors
                'weights': ['uniform', 'distance']
            }
        },
        'Support Vector Classifier': {
            'model': SVC(),
            'params': {
                'C': [0.1, 1],  # Reduced parameter space
                'kernel': ['linear', 'rbf']
            }
        }
    }

    best_results = {}
    for model_name, mp in param_grids.items():
        model = mp['model']
        param_grid = mp['params']

        # Use RandomizedSearchCV instead of GridSearchCV
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                           n_iter=5, cv=3, scoring='accuracy', verbose=1, random_state=42)
        random_search.fit(x_train_flattened, y_train_labels)

        print(f"\nBest parameters for {model_name}: {random_search.best_params_}")
        best_model = random_search.best_estimator_
        y_pred_grid = best_model.predict(x_test_flattened)
        accuracy_grid = accuracy_score(y_test_labels, y_pred_grid) * 100
        f1_grid = f1_score(y_test_labels, y_pred_grid, average='weighted') * 100
        recall_grid = recall_score(y_test_labels, y_pred_grid, average='weighted') * 100
        precision_grid = precision_score(y_test_labels, y_pred_grid, average='weighted') * 100
        print(f"\nBest accuracy for {model_name}: {accuracy_grid}%")
        print(f"\nBest f1 score for {model_name}: {f1_grid}%")
        print(f"\nBest recall score for {model_name}: {recall_grid}%")
        print(f"\nBest precision score for {model_name}: {precision_grid}%")
        print(f"Best Params {random_search.best_params_}")
        best_results[model_name] = {
            'Accuracy': accuracy_grid,
            'F1 Score': f1_grid,
            'Recall': recall_grid,
            'Precision': precision_grid
        }
        return accuracy_grid, f1_grid, recall_grid, precision_grid

def gradient_boosting(x_train_flattened, x_test_flattened, y_train_labels, y_test_labels):
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    xgb_model.fit(x_train_flattened, y_train_labels)
    y_pred_xgb = xgb_model.predict(x_test_flattened)
    accuracy_xgb = accuracy_score(y_test_labels, y_pred_xgb) * 100
    f1_xgb = f1_score(y_test_labels, y_pred_xgb, average='weighted') * 100
    recall_xgb = recall_score(y_test_labels, y_pred_xgb, average='weighted') * 100
    precision_xgb = precision_score(y_test_labels, y_pred_xgb, average='weighted') * 100
    return accuracy_xgb, f1_xgb, recall_xgb, precision_xgb

def stacking(x_train_flattened, x_test_flattened, y_train_labels, y_test_labels):
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('K-Nearest Neighbors', KNeighborsClassifier())
    ]
    meta_learner = SVC()
    stacking_model = StackingClassifier(estimators=models, final_estimator=meta_learner)
    stacking_model.fit(x_train_flattened, y_train_labels)
    y_pred_meta = stacking_model.predict(x_test_flattened)
    accuracy_meta = accuracy_score(y_test_labels, y_pred_meta) * 100
    f1_meta = f1_score(y_test_labels, y_pred_meta, average='weighted') * 100
    recall_meta = recall_score(y_test_labels, y_pred_meta, average='weighted') * 100
    precision_meta = precision_score(y_test_labels, y_pred_meta, average='weighted') * 100
    print(f"Stacking Model Accuracy: {accuracy_meta}")
    print(f"Stacking Model F1 Score: {f1_meta}")
    return accuracy_meta, f1_meta, recall_meta, precision_meta

base_path = "/Users/ehushubhamshaw/Desktop/KDD/assignment2/project2_kdd_items/archive"
train_directory = base_path + "/images_cancer_train"
validation_directory = base_path + "/images_cancer_test"

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(train_directory, target_size=(150, 150), class_mode='binary', batch_size=32)
validation_generator = datagen.flow_from_directory(validation_directory,target_size=(150, 150), class_mode='binary', batch_size=32)
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

if train_generator.samples > 0 and validation_generator.samples > 0:
    x_train, y_train = next(train_generator)
    x_test, y_test = next(validation_generator)
    x_train_flattened = x_train.reshape(x_train.shape[0], -1)  # Flatten for models
    x_test_flattened = x_test.reshape(x_test.shape[0], -1)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Classifier': SVC()
    }

    baseline_results = {}
    results_xgb = {}
    results_stacking = {}

    for name, model in models.items():
        model.fit(x_train_flattened, y_train)
        y_pred = model.predict(x_test_flattened)
        accuracy = accuracy_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        baseline_results[name] = {'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'Precision': precision}
        #accuracy_xgb, f1_xgb, recall_xgb, precision_xgb = gradient_boosting(x_train_flattened, x_test_flattened, y_train, y_test)
        #results_xgb[name] = {'Accuracy': accuracy_xgb, 'F1 Score': f1_xgb, 'Recall': recall_xgb,'Precision': precision_xgb}
        accuracy_meta, f1_meta, recall_meta, precision_meta = stacking(x_train_flattened, x_test_flattened, y_train, y_test)
        results_stacking[name] = {'Accuracy': accuracy_meta, 'F1 Score': f1_meta, 'Recall': recall_meta, 'Precision': precision_meta}

    #accuracy_grid, f1_grid, recall_grid, precision_grid = grid_search(x_train_flattened, y_train, x_test_flattened, y_test)
    #baseline_results['Grid'] = {'Accuracy' : accuracy_grid, 'F1 Score' : f1_grid, 'Recall' : recall_grid, 'Precision' : precision_grid}
    # Display the results
    # results_df = pd.DataFrame(baseline_results).T
    # print("Model Results:")
    # print(results_df)
    # results_df.plot(kind='bar')
    # plt.title('Model Performance Comparison')
    # plt.ylabel('Scores')
    # plt.xticks(rotation=45)
    # plt.show()

    # results_xgb_df = pd.DataFrame(results_xgb).T
    # print("Model Results:")
    # print(results_xgb_df)
    # results_xgb_df.plot(kind='bar')
    # plt.title('Model Performance Comparison xgb')
    # plt.ylabel('Scores')
    # plt.xticks(rotation=45)
    # plt.show()

    results_stacking_df = pd.DataFrame(results_stacking).T
    print("Model Results:")
    print(results_stacking_df)
    results_stacking_df.plot(kind='bar')
    plt.title('Model Performance Comparison stack')
    plt.ylabel('Scores')
    plt.xticks(rotation=45)
    plt.show()

else:
    print("No images found in the directories.")