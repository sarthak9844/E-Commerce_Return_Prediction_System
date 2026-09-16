import pandas as pd
import joblib
# After feature analysis
from feature_engineering import create_feature


# load cleaned dataset
df = pd.read_csv(r"D:\PROJECT\E-commerce _Return_prediction_system\Dataset\Cleaned_Ecommerce_Data.csv",encoding="utf-8")
df = create_feature(df)

x = df.drop('returned',axis=1)       # also use x = df.drop('returned',axis=1)
y = df['returned']

# features type
categorical_features = ["category","payment_method","region","customer_gender","age_group","delivery_group"]

numerical_features = ["price","discount","quantity","total_amount","shipping_cost","profit_margin","customer_age","delivery_days","order_year","order_month","order_day_of_week","price_per_quantity","amount_per_quantity","shipping_ratio","profit_amount"]

                    # train  test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42,stratify=y)


                    #create preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing  import OneHotEncoder,StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
       ("numerical",StandardScaler(),numerical_features),
       ("categorical",OneHotEncoder(handle_unknown="ignore"),categorical_features)
    ]
)

                    #create Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([("preprocessor",preprocessor),("model",LogisticRegression(max_iter=1000,random_state=42))])


                    #GridSearchCV
from sklearn.model_selection import GridSearchCV

parameter ={
    'model__C':[0.01, 0.1, 1, 10, 100],'model__class_weight':['balanced',None], 'model__solver':['lbfgs','liblinear']}

logistic_model = GridSearchCV(estimator=pipeline,param_grid=parameter,cv=5,scoring="f1",n_jobs=-1,verbose=1)


#train the model
logistic_model.fit(x_train,y_train)
print(logistic_model.best_params_)          # best-parameter   
print(logistic_model.best_score_)           # best-validatin-score
best_model = logistic_model.best_estimator_      # best model   

y_pred = best_model.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,f1_score
print(logistic_model.score(x_train,y_train))
print(classification_report(y_test,y_pred)) 
print(accuracy_score(y_test,y_pred)) 
print(f1_score(y_test,y_pred))
                #save the model
model_path = r"model\logistic_regression_model.pkl"
joblib.dump(
    best_model,
    model_path
)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print(confusion_matrix(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, f1_score
import numpy as np

                # Training data ke out-of-fold probabilities
train_proba = cross_val_predict(best_model,x_train,y_train,cv=5,method="predict_proba",n_jobs=-1
)[:, 1]


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, f1_score
import numpy as np

# Training data ke out-of-fold probabilities
train_proba = cross_val_predict(best_model,x_train,y_train,cv=5,
method="predict_proba",n_jobs=-1
)[:, 1]

# Different thresholds try karna
thresholds = np.arange(0.10, 0.51, 0.01)
f1_scores = []
for threshold in thresholds:
    y_train_pred = (train_proba >= threshold).astype(int)
    score = f1_score(y_train, y_train_pred)
    f1_scores.append(score)

                    # Best threshold find karna
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_validation_f1 = f1_scores[best_index]


print("Best Threshold:", best_threshold)
print("Best Validation F1:", best_validation_f1)


                    # Test data probabilities
test_proba = best_model.predict_proba(x_test)[:, 1]
                    # Selected threshold se prediction
y_test_pred = (test_proba >= best_threshold).astype(int)

                    # Final evaluation
print("\nFinal Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Final Test F1:")
print(f1_score(y_test, y_test_pred))