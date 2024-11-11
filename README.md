# EX.NO.09        A project on Time series analysis on Electric Production Data using ARIMA model 
### Date: 
### Name:A.Sasidharan
### Register no: 212221240049

### AIM:
To Create a project on Time series analysis on Global Temperature Data using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # This line is added to import the necessary plotting functions

data = pd.read_csv("/content/globaltemper.csv")
data.head()
# data.plot(figsize=(10,5)) #Commented out as 'IPG2211A2N' might not be a valid column for plotting
# plt.title("Electric Production Over Time") #Commented out to avoid potential error with 'IPG2211A2N'
# plt.show() 

# Assuming 'AverageTemperature' or 'LandAverageTemperature' is the relevant column for your analysis
target_variable = 'AverageTemperature'  # Or 'LandAverageTemperature' if that's the correct column name

def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    return result[1] < 0.05  # True if stationary #Fixed: Indentation corrected to align with function body
# Replace 'IPG2211A2N' with the actual column name from your dataset
is_stationary = adf_test(data[target_variable]) 

if not is_stationary:
    data_diff = data[target_variable].diff().dropna()
    plt.plot(data_diff)
    plt.title("Differenced Global Temperature") # Title adjusted to reflect the target variable
    plt.show()
else:
    data_diff = data[target_variable]

plot_acf(data_diff, lags=20) # Assuming 'plot_acf' is imported from statsmodels.graphics.tsaplots
plt.show()

plot_pacf(data_diff, lags=12) # Assuming 'plot_pacf' is imported from statsmodels.graphics.tsaplots
plt.show()

p, d, q = 1, 1, 1  # example values; adjust based on plots

# Replace 'IPG2211A2N' with the actual column name from your dataset
model = ARIMA(data[target_variable], order=(p, d, q))  
fitted_model = model.fit()
print(fitted_model.summary())

forecast_steps = 12  # Number of months to forecast
forecast = fitted_model.forecast(steps=forecast_steps)

# Assuming 'dt' is your date column or replace with the appropriate date column
last_date = pd.to_datetime(data['dt'].iloc[-1])  # Convert to datetime if necessary
forecast_index = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS') 

# Replace 'IPG2211A2N' with the actual column name from your dataset
plt.plot(data[target_variable], label="Historical Data")  #Fixed: Plot data["IPG2211A2N"]
plt.plot(forecast_index, forecast, label="Forecast", color='orange')
plt.legend()
plt.title("Global Temperature Forecast") # Title adjusted to reflect the target variable
plt.show()
~~~
### OUTPUT:

### AutoCorrelation

![image](https://github.com/user-attachments/assets/449aff92-3668-49f4-a1eb-44418be14ea0)

### Partial Autocorrelation
![image](https://github.com/user-attachments/assets/7e9b0f31-a209-4114-88b1-dac76057a468)

### Model Results
![image](https://github.com/user-attachments/assets/e5ef0271-3612-46cc-bedd-c13e8d7cc0e0)

### Electric Production Forecast
![image](https://github.com/user-attachments/assets/a2d026ad-63a8-4433-a85e-40e1684f8efb)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
