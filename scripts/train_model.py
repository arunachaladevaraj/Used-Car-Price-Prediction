from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from warnings import simplefilter
import pickle

# Model Training and Evaluation
features = ['transmission', 'Year of Manufacture', 'modelYear', 'City', 'Insurance Validity', 'Ownership', 
            'fuelType', 'kilometersDriven', 'bodyType', 'Mileage']
target = 'price'

X = df[features]
y = df[target]

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Year of Manufacture', 'modelYear', 'kilometersDriven', 'Mileage']),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), 
         ['transmission', 'City', 'Ownership', 'fuelType', 'bodyType', 'Insurance Validity'])
    ])

# Define models and their hyperparameters
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                         ('model', LinearRegression())]),
    'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', DecisionTreeRegressor())]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', RandomForestRegressor())]),
    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor),
                                         ('model', GradientBoostingRegressor())])
}

# Hyperparameters for Grid Search
param_grids = {
    'Decision Tree': {'model__max_depth': [5, 10, 15, None]},
    'Random Forest': {'model__n_estimators': [50, 100, 150],
                      'model__max_depth': [5, 10, 15, None]},
    'Gradient Boosting': {'model__n_estimators': [50, 100, 150],
                          'model__learning_rate': [0.01, 0.1, 0.2]}
}

# Train and evaluate models
best_models = {}
for name, pipeline in models.items():
    if name in param_grids:
        grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, n_jobs=-1, scoring='r2')
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"{name}: Best parameters: {grid_search.best_params_}")
    else:
        pipeline.fit(X_train, y_train)
        best_models[name] = pipeline
    
    y_pred = best_models[name].predict(X_test)
    print(f"\n{name} Evaluation:")
    print(f"R2 Score: {r2_score(y_test, y_pred)}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")


import pickle

# Assuming the Random Forest model is stored in 'best_models' under the key 'Random Forest'
random_forest_model = best_models['Random Forest']

# Specify the file path where you want to save the model
pickle_file_path = 'C:\\Users\\arund\\Downloads\\random_forest_model.pkl'

# Save the model to a pickle file
with open(pickle_file_path, 'wb') as file:
    pickle.dump(random_forest_model, file)

print(f"Random Forest model saved to {pickle_file_path}")
