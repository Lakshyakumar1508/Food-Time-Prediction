from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create your views here.
def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    # Load the dataset
    data = pd.read_csv(r"D:\Coding Space\Django\FoodTimePrediction\Food_Delivery_Time_Prediction.csv")

    # Encode categorical features
    le = LabelEncoder()
    for col in data.select_dtypes(include="object").columns:
        data[col] = le.fit_transform(data[col])

    # Standardizing numerical features
    scaler = StandardScaler()
    data[["Distance", "Delivery_Time", "Order_Cost"]] = scaler.fit_transform(data[["Distance", "Delivery_Time", "Order_Cost"]])

    # Dropping unnecessary columns
    data.drop(columns=["Order_ID", "Customer_Rating", "Restaurant_Rating"], inplace=True)

    # Feature Engineering
    data['Order_Time'] = data['Order_Time'].astype(str).str.zfill(4)
    data['Rush_Hour'] = data['Order_Time'].apply(lambda x: 1 if 7 <= int(x[:2]) <= 9 or 17 <= int(x[:2]) <= 20 else 0)

    # Target variable
    data["Delivery_Status"] = data["Delivery_Time"].apply(lambda x: 1 if x > 0 else 0)  # Ensure it has both classes

    # Debug: Check class distribution
    print("Unique values in Delivery_Status:", data["Delivery_Status"].value_counts())

    # Define features and target
    X = data[["Distance", "Traffic_Conditions", "Weather_Conditions", "Delivery_Person_Experience", "Rush_Hour", "Order_Priority"]]
    y = data["Delivery_Status"]

    # Split the dataset with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Get user input from request
    try:
        var1 = float(request.GET['n1'])
        var2 = float(request.GET['n2'])
        var3 = float(request.GET['n3'])
        var4 = float(request.GET['n4'])
        var5 = float(request.GET['n5'])
        var6 = float(request.GET['n6'])
    except KeyError:
        return render(request, "predict.html", {"result2": "Invalid input, please provide all values."})

    # Make prediction
    prediction = model.predict(np.array([var1, var2, var3, var4, var5, var6]).reshape(1, -1))
    prediction = int(prediction[0])

    # Convert prediction to meaningful output
    result_text = "The Predicted Delivery Status is: " + ("Delayed" if prediction == 1 else "On Time")

    return render(request, "predict.html", {"result2": result_text})
