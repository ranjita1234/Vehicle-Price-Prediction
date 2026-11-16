from flask import Flask, render_template, request , jsonify
from vehicles.car import carPriceModel
from vehicles.bike import bikePriceModel
from vehicles.bus import busPriceModel
from vehicles.truck import truckPriceModel

# Initialize the Flask app
app = Flask(__name__)

car_model = carPriceModel()
bike_model = bikePriceModel()
bus_model = busPriceModel()
truck_model = truckPriceModel()

# Define a route for the homepage
@app.route("/")
def home():
    return render_template("./index.html")

@app.route("/about")
def about():
    return render_template("./views/about.html")

@app.route("/contactus")
def contactus():
    return render_template("./views/contactus.html")

@app.route("/analytics")
def analytics():
    return render_template("./views/analytics.html")

@app.route("/privacypolicy")
def policy():
    return render_template("./views/privacypolicies.html")

warnings = {}
@app.route('/car', methods=['GET' ,'POST'])
def carResult():
    if request.method == 'POST':
        car_name = request.form['car_name']
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        driven_kms = float(request.form['driven_kms'])
        fuel_type = request.form['fuel_type']
        selling_type = request.form['selling_type']
        transmission = request.form['transmission']
        owner = int(request.form['owner'])


        car_name_encoded = car_model.Car_Name_le.transform([car_name])[0]
        fuel_type_encoded = car_model.Fuel_Type_le.transform([fuel_type])[0]
        selling_type_encoded = car_model.Selling_type_le.transform([selling_type])[0]
        transmission_encoded = car_model.Transmission_le.transform([transmission])[0]

        new_data_point = [[car_name_encoded, year, present_price, driven_kms, fuel_type_encoded, selling_type_encoded, transmission_encoded, owner]]

        new_data_point_standardized = car_model.ss.transform(new_data_point)

        prediction = car_model.rf.predict(new_data_point_standardized)
        result = round(prediction.item(), 4)

        return render_template('./vehicles/car.html', prediction=result,warnings=warnings, Data = [car_name,year,present_price,driven_kms,fuel_type,selling_type,transmission]) 

    else:
        return render_template("./vehicles/car.html")
    
@app.route("/bus", methods=['GET','POST'])
def busResult():
    if request.method == 'POST':
        Bus_Name = request.form['Bus_Name']
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        driven_kms = float(request.form['driven_kms'])
        fuel_type = request.form['fuel_type']
        selling_type =  request.form['selling_type']
        transmission = request.form['transmission']
        owner = int(request.form['owner'])

        Bus_name_encoded = bus_model.Bus_Name_le.transform([Bus_Name])[0]
        fuel_type_encoded = bus_model.Fuel_Type_le.transform([fuel_type])[0]
        selling_type_encoded = bus_model.Selling_type_le.transform([selling_type])[0]
        transmission_encoded = bus_model.Transmission_le.transform([transmission])[0]

        new_data_point = [[Bus_name_encoded, year, present_price, driven_kms, fuel_type_encoded, selling_type_encoded, transmission_encoded, owner]]

        new_data_point_standardized = bus_model.ss.transform(new_data_point)

        prediction = bus_model.rf.predict(new_data_point_standardized)

        result = round(prediction.item(), 4)

        return render_template("./vehicles/bus.html", prediction = result, Data = [Bus_Name,year,present_price,driven_kms,fuel_type,selling_type,transmission])
    else:
        return render_template("./vehicles/bus.html")



@app.route("/bike", methods=['GET' ,'POST'])
def bikeResult():
    if request.method == 'POST':
        Bike_name = request.form['Bike_name']
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        driven_kms = float(request.form['driven_kms'])
        fuel_type="Petrol"
        selling_type =  request.form['selling_type']
        transmission="Manual"
        owner = int(request.form['owner'])

        Bike_name_encoded = bike_model.Bike_Name_le.transform([Bike_name])[0]
        fuel_type_encoded = bike_model.Fuel_Type_le.transform([fuel_type])[0]
        selling_type_encoded = bike_model.Selling_type_le.transform([selling_type])[0]
        transmission_encoded = bike_model.Transmission_le.transform([transmission])[0]

        new_data_point = [[Bike_name_encoded, year, present_price, driven_kms, fuel_type_encoded, selling_type_encoded, transmission_encoded, owner]]

        new_data_point_standardized = bike_model.ss.transform(new_data_point)

        prediction = bike_model.rf.predict(new_data_point_standardized)

        result = round(prediction.item(), 4)

        return render_template("./vehicles/bike.html", prediction = result, Data = [Bike_name,year,present_price,driven_kms,fuel_type,selling_type,transmission])
    else:
        return render_template("./vehicles/bike.html")


@app.route("/truck", methods=['GET', 'POST'])
def truckResult():
    if request.method == 'POST':
        Truck_Name = request.form['Truck_Name']
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        driven_kms = float(request.form['driven_kms'])
        fuel_type = "Diesel"
        selling_type = request.form['selling_type']
        transmission = request.form['transmission']
        owner = int(request.form['owner'])

        Truck_name_encoded = truck_model.Truck_Name_le.transform([Truck_Name])[0]
        fuel_type_encoded = truck_model.Fuel_Type_le.transform([fuel_type])[0]
        selling_type_encoded = truck_model.Selling_type_le.transform([selling_type])[0]
        transmission_encoded = truck_model.Transmission_le.transform([transmission])[0]


        new_data_point = [[Truck_name_encoded, year, present_price, driven_kms, fuel_type_encoded, selling_type_encoded, transmission_encoded, owner]]

        new_data_point_standardized = truck_model.ss.transform(new_data_point)


        prediction = truck_model.rf.predict(new_data_point_standardized)

        result = round(prediction.item(), 4)

        return render_template("./vehicles/truck.html", prediction = result, Data = [Truck_Name,year,present_price,driven_kms,fuel_type,selling_type,transmission])
    else:
        return render_template("./vehicles/truck.html")

# Run the app
if __name__ == "__main__":
    app.run(port=8200,debug=True)
