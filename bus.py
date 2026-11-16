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
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error,root_mean_squared_error


class busPriceModel:

    dataset = pd.read_csv("data/Bus_Data.csv")

    dataset.isnull().sum()

    min_year = dataset['Year'].min()
    max_year = dataset['Year'].max()

    plt.figure(figsize=(10, 6))
    plt.hist(dataset['Year'], bins=range(dataset['Year'].min(), dataset['Year'].max() + 1), color='skyblue', edgecolor='black')
    plt.title('Distribution of bus by Year')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("static/img/bus/distribution_of_bus_by_year.png")


    max_selling_bus = dataset.loc[dataset['Selling_Price'].idxmax(), 'Bus_Name']
    min_selling_bus = dataset.loc[dataset['Selling_Price'].idxmin(), 'Bus_Name']
    max_selling_price = dataset['Selling_Price'].max()
    min_selling_price = dataset['Selling_Price'].min()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh([max_selling_bus, min_selling_bus], [dataset['Selling_Price'].max(), dataset['Selling_Price'].min()], color=['green', 'red'])
    plt.xlabel('Selling Price')
    plt.title('Bus with Maximum and Minimum Selling Prices')
    plt.grid(axis='x')
    plt.savefig("static/img/bus/bus_max_min_sell.png")
    plt.close()

    fuel_type_counts = dataset['Fuel_Type'].value_counts()

    # Plotting
    plt.pie(fuel_type_counts, labels=fuel_type_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Distribution of bus based on Fuel Type')
    plt.savefig("static/img/bus/distribution_of_bus_by_fuel_type.png")
    plt.close()

    selling_type_counts = dataset['Selling_type'].value_counts()

    # Plot the counts
    plt.bar(selling_type_counts.index, selling_type_counts.values, color=['blue', 'brown'])
    plt.title('Distribution of Selling Types')
    plt.xlabel('Selling Type')
    plt.ylabel('Number of Bus Sold')
    plt.savefig("static/img/bus/distribution_of_bus_by_sell_type.png")
    plt.close()

    transmission_counts = dataset['Transmission'].value_counts()

    # Plotting the transmission types as a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(transmission_counts, labels=transmission_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Transmission Types')
    plt.axis('equal')
    plt.savefig("static/img/bus/bus_transmission_type.png")
    plt.close()

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
    plt.savefig("static/img/bus/avg_sell_bus.png")
    plt.close()

    # Calculate correlation
    correlation = dataset['Selling_Price'].corr(dataset['Driven_kms'])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Driven_kms", y="Selling_Price", data=dataset, hue="Fuel_Type", style="Transmission", palette="viridis")
    plt.title("Relationship between Selling Price and Mileage")
    plt.xlabel("Driven Kilometers")
    plt.ylabel("Selling Price (in Lakhs)")
    plt.legend(title="Fuel Type")
    plt.grid(True)
    plt.savefig("static/img/bus/rel_btw_price_mileage.png")
    plt.close()

    correlation = dataset['Selling_Price'].corr(dataset['Owner'])

    plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    plt.scatter(dataset['Owner'], dataset['Selling_Price'], color='blue')
    plt.title('Relationship between Selling Price and Owner')
    plt.xlabel('Owner')
    plt.ylabel('Selling Price (in Lakhs)')
    plt.xticks(range(max(dataset['Owner']) + 1))
    plt.grid(True)
    plt.savefig("static/img/bus/rel_btw_price_owner.png")
    plt.close()

    correlation = dataset['Selling_Price'].corr(dataset['Year'])

    plt.figure(figsize=(14, 6))
    sns.scatterplot(x="Year", y="Selling_Price", data=dataset, hue="Bus_Name", palette="viridis")
    plt.title('Selling Price vs Years')
    plt.xlabel('Year')
    plt.ylabel('Selling Price')
    plt.grid(True)
    plt.savefig("static/img/bus/price_vs_years.png")
    plt.close()

    brand_distribution = dataset['Bus_Name'].value_counts()

    top_10_brands = brand_distribution.head(10)

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(top_10_brands, labels=top_10_brands.index, autopct='%1.1f%%', startangle=140)
    plt.title('Top 10 Vehicle Brands Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("static/img/bus/top_10_bus.png")
    plt.close()

    Bus_Name_le = LabelEncoder()
    dataset['Bus_Name'] = Bus_Name_le.fit_transform(dataset['Bus_Name'])

    Fuel_Type_le = LabelEncoder()
    dataset['Fuel_Type'] = Fuel_Type_le.fit_transform(dataset['Fuel_Type'])

    Selling_type_le = LabelEncoder()
    dataset['Selling_type'] = Selling_type_le.fit_transform(dataset['Selling_type'])

    Transmission_le = LabelEncoder()
    dataset['Transmission'] = Transmission_le.fit_transform(dataset['Transmission'])

    input_data = dataset[["Bus_Name","Year","Present_Price", "Driven_kms","Fuel_Type", "Selling_type", "Transmission", "Owner" ]]
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


    # Train and evaluate Support Vector Regressor (SVR)
    svr = SVR() # Initialize the SVR model
    svr.fit(x_train, y_train)
    y_train_pred_svr = svr.predict(x_train)
    y_test_pred_svr = svr.predict(x_test)
    train_rmse_svr = root_mean_squared_error(y_train, y_train_pred_svr) # Calculate train RMSE for SVR
    test_rmse_svr = root_mean_squared_error(y_test, y_test_pred_svr)   # Calculate test RMSE for SVR

    # Compare models based on test set performance
    models_performance = {
        "Decision Tree": (test_rmse_dt, dt.score(x_test,y_test)*100),
        "Random Forest": (test_rmse_rf, rf.score(x_test,y_test)*100),
        "Linear Regression": (test_rmse_lr, lr.score(x_test,y_test)*100),
        "Support Vector Regressor": (test_rmse_svr, svr.score(x_test,y_test)*100)
    }

    # Sort models based on RMSE (lower is better) and R squared (higher is better)
    sorted_models = sorted(models_performance.items(), key=lambda x: (x[1][0], x[1][1]), reverse=False)

    # Print the sorted list
    # print("Models sorted by RMSE and R squared (Test Set):")
    # for model, (rmse, r2) in sorted_models:
    #     print(f"{model}: RMSE = {rmse}, R squared = {r2}")

    # Choose the best model
    best_model_name, (best_rmse, best_r2) = sorted_models[0]


    new_data = {
        "Bus_Name": "Ashok Leyland Circuit",
        "Year": 2006,
        "Present_Price": 36.41,
        "Driven_kms": 449209,
        "Fuel_Type": "Diesel",
        "Selling_type": "Dealer",
        "Transmission": "Manual",
        "Owner": 1
    }



    new_data["Bus_Name"] = Bus_Name_le.transform([new_data["Bus_Name"]])[0]
    new_data["Fuel_Type"] = Fuel_Type_le.transform([new_data["Fuel_Type"]])[0]
    new_data["Selling_type"] = Selling_type_le.transform([new_data["Selling_type"]])[0]
    new_data["Transmission"] = Transmission_le.transform([new_data["Transmission"]])[0]
    new_data_df = pd.DataFrame([new_data])

    new_data_standardized = pd.DataFrame(ss.transform(new_data_df), columns=new_data_df.columns)

    # Make predictions on the standardized new data using the trained Random Forest model
    predicted_price = rf.predict(new_data_standardized)


    # Modify new_data to use original string values
    new_data = {
        "Bus_Name": "Ashok Leyland Circuit",
        "Year": 2006,
        "Present_Price": 36.41,
        "Driven_kms": 449209,
        "Fuel_Type": "Diesel",
        "Selling_type": "Dealer",
        "Transmission": "Manual",
        "Owner": 1
    }

    # Now you can use the fitted LabelEncoders to transform your new data
    new_data["Bus_Name"] = Bus_Name_le.transform([new_data["Bus_Name"]])[0]
    new_data["Fuel_Type"] = Fuel_Type_le.transform([new_data["Fuel_Type"]])[0] 
    new_data["Selling_type"] = Selling_type_le.transform([new_data["Selling_type"]])[0] 
    new_data["Transmission"] = Transmission_le.transform([new_data["Transmission"]])[0] 
    new_data_df = pd.DataFrame([new_data])

    predicted_price = rf.predict(new_data_standardized)


