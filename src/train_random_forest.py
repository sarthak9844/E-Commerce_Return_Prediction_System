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

                # create the pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ("preprocessor",preprocessor),
    ("model",RandomForestClassifier(random_state=42))
])

                # Prameter and GridsearchCV
from sklearn.model_selection import GridSearchCV

parameter ={
    "model__n_estimators": [100,200],"model__max_depth": [5,10,None],"model__min_samples_split": [2,5],"model__min_samples_leaf": [1,2],"model__class_weight": ["balanced"]
}

random_model = GridSearchCV(
    estimator=pipeline,param_grid=parameter,cv=5,scoring="f1",n_jobs=-1,verbose=1
)

random_model.fit(x_train,y_train)
print(random_model.best_params_)               # best-parameter   
print(random_model.best_score_)                # best-validatin-score
best_model = random_model.best_estimator_      # best model   

y_pred = best_model.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,f1_score

print(classification_report(y_test,y_pred)) 
print(accuracy_score(y_test,y_pred)) 
print(f1_score(y_test,y_pred))

                #save the model
model_path = r"model\Random_tree_model.pkl"
joblib.dump(best_model,model_path)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print(confusion_matrix(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()