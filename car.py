import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, root_mean_squared_error

import os

class carPriceModel:
    dataset = pd.read_csv("data/Car_Data.csv")

    dataset.isnull().sum()

    min_year = dataset['Year'].min()
    max_year = dataset['Year'].max()

    plt.figure(figsize=(10, 6))
    plt.hist(dataset['Year'], bins=range(dataset['Year'].min(), dataset['Year'].max() + 1), color='skyblue', edgecolor='black')
    plt.title('Distribution of Cars by Year')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("static/img/car/distribution_of_cars_by_year.png")

    max_selling_car = dataset.loc[dataset['Selling_Price'].idxmax(), 'Car_Name']
    min_selling_car = dataset.loc[dataset['Selling_Price'].idxmin(), 'Car_Name']
    max_selling_price = dataset['Selling_Price'].max()
    min_selling_price = dataset['Selling_Price'].min()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh([max_selling_car, min_selling_car], [dataset['Selling_Price'].max(), dataset['Selling_Price'].min()], color=['green', 'red'])
    plt.xlabel('Selling Price')
    plt.title('Cars with Maximum and Minimum Selling Prices')
    plt.grid(axis='x')
    plt.savefig("static/img/car/max_min_sell_price.png")

    fuel_type_counts = dataset['Fuel_Type'].value_counts()

    # Plotting
    plt.pie(fuel_type_counts, labels=fuel_type_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Distribution of Cars based on Fuel Type')
    plt.savefig("static/img/car/distribution_of_cars_by_fuel.png")



    selling_type_counts = dataset['Selling_type'].value_counts()

    # Plot the counts
    plt.bar(selling_type_counts.index, selling_type_counts.values, color=['blue', 'brown'])
    plt.title('Distribution of Selling Types')
    plt.xlabel('Selling Type')
    plt.ylabel('Number of Cars Sold')
    plt.savefig("static/img/car/distribution_of_cars_by_selltype.png")

    transmission_counts = dataset['Transmission'].value_counts()

    # Plotting the transmission types as a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(transmission_counts, labels=transmission_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Transmission Types')
    plt.axis('equal')
    plt.savefig("static/img/car/transmission_type.png")



    price_trends = dataset.groupby('Year')['Selling_Price'].mean().reset_index()

    # Sorting the DataFrame by year
    price_trends = price_trends.sort_values(by='Year')

    # Plotting the price trends
    plt.figure(figsize=(10, 6))
    plt.plot(price_trends['Year'], price_trends['Selling_Price'], marker='o', linestyle='-')
    plt.title('Average Selling Price Trends Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Average Selling Price')
    plt.grid(True)
    plt.xticks(price_trends['Year'])
    plt.tight_layout()
    plt.savefig("static/img/car/avg_sell_trends.png")



    # Calculate correlation
    correlation = dataset['Selling_Price'].corr(dataset['Driven_kms'])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Driven_kms", y="Selling_Price", data=dataset, hue="Fuel_Type", style="Transmission", palette="viridis")
    plt.title("Relationship between Selling Price and Mileage")
    plt.xlabel("Driven Kilometers")
    plt.ylabel("Selling Price (in Lakhs)")
    plt.legend(title="Fuel Type")
    plt.grid(True)
    plt.savefig("static/img/car/rel_bw_price_mileage.png")


    correlation = dataset['Selling_Price'].corr(dataset['Owner'])

    plt.figure(figsize=(10, 6))
    plt.scatter(dataset['Owner'], dataset['Selling_Price'], color='blue')
    plt.title('Relationship between Selling Price and Owner')
    plt.xlabel('Owner')
    plt.ylabel('Selling Price (in Lakhs)')
    plt.xticks(range(max(dataset['Owner']) + 1))
    plt.grid(True)
    plt.savefig("static/img/car/rel_bw_price_owner.png")


    correlation = dataset['Selling_Price'].corr(dataset['Year'])

    plt.figure(figsize=(14, 6))
    sns.scatterplot(x="Year", y="Selling_Price", data=dataset,hue="Car_Name", palette="viridis")
    plt.title('Selling Price vs Years')
    plt.xlabel('Year')
    plt.ylabel('Selling Price')
    plt.grid(True)
    plt.savefig("static/img/car/price_vs_years.png")


    brand_distribution = dataset['Car_Name'].value_counts()

    top_10_brands = brand_distribution.head(10)

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(top_10_brands, labels=top_10_brands.index, autopct='%1.1f%%', startangle=140)
    plt.title('Top 10 Vehicle Brands Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("static/img/car/top_brands.png")
    plt.close()


    Car_Name_le = LabelEncoder()
    dataset['Car_Name'] = Car_Name_le.fit_transform(dataset['Car_Name'])

    Fuel_Type_le = LabelEncoder()
    dataset['Fuel_Type'] = Fuel_Type_le.fit_transform(dataset['Fuel_Type'])

    Selling_type_le = LabelEncoder()
    dataset['Selling_type'] = Selling_type_le.fit_transform(dataset['Selling_type'])

    Transmission_le = LabelEncoder()
    dataset['Transmission'] = Transmission_le.fit_transform(dataset['Transmission'])


    input_data = dataset[["Car_Name","Year","Present_Price", "Driven_kms","Fuel_Type", "Selling_type", "Transmission", "Owner" ]]
    output_data = dataset["Selling_Price"]



    ss = StandardScaler()
    input_data  = pd.DataFrame(ss.fit_transform(input_data),columns = input_data.columns)

    x_train,x_test,y_train,y_test = train_test_split(input_data, output_data , test_size = 0.2, random_state = 42)



    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    train_rmse_lr = root_mean_squared_error(y_train, y_train_pred)
    test_rmse_lr = root_mean_squared_error(y_test, y_test_pred)


    rf = RandomForestRegressor(random_state=42)
    rf.fit(x_train, y_train)
    y_train_pred = rf.predict(x_train)
    y_test_pred = rf.predict(x_test)
    train_rmse_rf = root_mean_squared_error(y_train, y_train_pred)
    test_rmse_rf = root_mean_squared_error(y_test, y_test_pred)


    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(x_train, y_train)
    y_train_pred = dt.predict(x_train)
    y_test_pred = dt.predict(x_test)
    train_rmse_dt = root_mean_squared_error(y_train, y_train_pred)
    test_rmse_dt = root_mean_squared_error(y_test, y_test_pred)


    svr = SVR()
    svr.fit(x_train, y_train)
    y_train_pred = svr.predict(x_train)
    y_test_pred = svr.predict(x_test)
    train_rmse_svr = root_mean_squared_error(y_train, y_train_pred)
    test_rmse_svr = root_mean_squared_error(y_test, y_test_pred)


    # Compare models based on test set performance
    models_performance = {
        "Decision Tree": (test_rmse_dt, dt.score(x_test,y_test)*100),
        "Random Forest": (test_rmse_rf, rf.score(x_test,y_test)*100),
        "Linear Regression": (test_rmse_lr, lr.score(x_test,y_test)*100),
        "Support Vector Regressor": (test_rmse_svr, svr.score(x_test,y_test)*100)
    }

    # Sort models based on RMSE (lower is better) and R squared (higher is better)
    sorted_models = sorted(models_performance.items(), key=lambda x: (x[1][0], x[1][1]), reverse=False)

    # for model, (rmse, r2) in sorted_models:
    #     print(f"{model}: RMSE = {rmse}, R squared = {r2}")

    # Choose the best model
    best_model_name, (best_rmse, best_r2) = sorted_models[0]


    new_data = {
        "Car_Name": "ritz",
        "Year": 2014,
        "Present_Price": 5.59,
        "Driven_kms": 27000,
        "Fuel_Type": "Petrol",
        "Selling_type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0
    }



    new_data["Car_Name"] = Car_Name_le.transform([new_data["Car_Name"]])[0]
    new_data["Fuel_Type"] = Fuel_Type_le.transform([new_data["Fuel_Type"]])[0]
    new_data["Selling_type"] = Selling_type_le.transform([new_data["Selling_type"]])[0]
    new_data["Transmission"] = Transmission_le.transform([new_data["Transmission"]])[0]
    new_data_df = pd.DataFrame([new_data])



    new_data_standardized = pd.DataFrame(ss.transform(new_data_df), columns=new_data_df.columns)

    # Make predictions on the standardized new data using the trained Random Forest model
    predicted_price = rf.predict(new_data_standardized)


    new_data = {
        "Car_Name": "city",  # Example car name
        "Year": 2019,  # Example year
        "Present_Price": 10.0,  # Example present price
        "Driven_kms": 5000,  # Example driven kilometers
        "Fuel_Type": "Petrol",  # Example fuel type
        "Selling_type": "Dealer",  # Example selling type
        "Transmission": "Manual",  # Example transmission type
        "Owner": 0  # Example owner
    }


    new_data["Car_Name"] = Car_Name_le.transform([new_data["Car_Name"]])[0]
    new_data["Fuel_Type"] = Fuel_Type_le.transform([new_data["Fuel_Type"]])[0]
    new_data["Selling_type"] = Selling_type_le.transform([new_data["Selling_type"]])[0]
    new_data["Transmission"] = Transmission_le.transform([new_data["Transmission"]])[0]
    new_data_df = pd.DataFrame([new_data])


    new_data_standardized = pd.DataFrame(ss.transform(new_data_df), columns=new_data_df.columns)

    # Make predictions on the standardized new data using the trained Random Forest model
    predicted_price = rf.predict(new_data_standardized)

















