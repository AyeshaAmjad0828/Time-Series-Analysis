import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

# Read data from Excel file
file_path = 'data/MetalPrices.xlsx'  
df = pd.read_excel(file_path)

colors = ['blue', 'red', 'green', 'purple']

# Create traces for each metal
traces = []
for i, metal in enumerate(df.columns[1:]):  
    trace = go.Scatter(
        x=df['Date'],
        y=df[metal],
        mode='lines',
        name=metal,
        line=dict(color=colors[i])  # Assigning a color to each trace
    )
    traces.append(trace)

# Layout for the chart
layout = go.Layout(
    title='Prices of Metals Over Time',
    xaxis=dict(title='Month'),
    yaxis=dict(title='Price')
)

# Create figure with traces and layout
fig = go.Figure(data=traces, layout=layout)

# Display the chart
pyo.plot(fig, filename='metals_prices.html')

# Create figure with traces
fig = make_subplots(specs=[[{"secondary_y": True}]])
for trace in traces:
    fig.add_trace(trace, secondary_y=False)

# Update layout for the chart
fig.update_layout(
    title='Prices of Metals Over Time',
    xaxis=dict(title='Date'),  # You might want to label this as 'Date' for monthly data
    yaxis=dict(title='Price')
)

##Decomposition
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform time series decomposition for each metal
decomposition = {}
for metal in df.columns[1:]:
    result = seasonal_decompose(df[metal], model='additive', period=12)  # Assuming monthly data with a yearly seasonality
    decomposition[metal] = result

# Plot the decomposed components for each metal
for metal, result in decomposition.items():
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    fig.suptitle(f"Decomposition of {metal} Prices")

    axes[0].set_title('Observed')
    axes[0].plot(df[metal], label='Original', color='blue')
    axes[0].legend()

    axes[1].set_title('Trend')
    axes[1].plot(result.trend, label='Trend', color='orange')
    axes[1].legend()

    axes[2].set_title('Seasonal')
    axes[2].plot(result.seasonal, label='Seasonal', color='green')
    axes[2].legend()

    axes[3].set_title('Residual')
    axes[3].plot(result.resid, label='Residual', color='red')
    axes[3].legend()

    plt.tight_layout()
    plt.show()


##Stationary Test
from statsmodels.tsa.stattools import adfuller

# Perform ADF test for each metal
adf_results = {}
for metal in df.columns[1:]:
    result = adfuller(df[metal], autolag='AIC')  # AIC is used to automatically select the lag
    adf_results[metal] = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values (1%)': result[4]['1%'],
        'Critical Values (5%)': result[4]['5%'],
    }

# Create a DataFrame to display ADF test results
adf_df = pd.DataFrame(adf_results).T
adf_df.index.name = 'Metal'

# Display the ADF test results table
print(adf_df)

##Performing first-order differencing

metal_prices = df.iloc[:, 1:]  
differenced_metal_prices = metal_prices.diff().dropna()

# Concatenate the differenced metal prices with the date field
differenced_df_with_date = pd.concat([df.iloc[1:, 0], differenced_metal_prices], axis=1)

# Display the first few rows of the concatenated data
print(differenced_df_with_date.head())

# Perform ADF test for each metal again after differencing
adf_results = {}
for metal in differenced_df_with_date.columns[1:]:
    result = adfuller(differenced_df_with_date[metal], autolag='AIC')  # AIC is used to automatically select the lag
    adf_results[metal] = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values (1%)': result[4]['1%'],
        'Critical Values (5%)': result[4]['5%'],
    }

# Create a DataFrame to display ADF test results
adf_df = pd.DataFrame(adf_results).T
adf_df.index.name = 'Metal'

# Display the ADF test results table
print(adf_df)

# Plot the differenced metal fields
plt.figure(figsize=(10, 6))
for column in differenced_df_with_date.columns[1:]:
    plt.plot(differenced_df_with_date.iloc[:, 0], differenced_df_with_date[column], label=column)

plt.title('Differenced Metal Prices')
plt.xlabel('Date')
plt.ylabel('Differenced Price')
plt.legend()
plt.tight_layout()
plt.show()

# Plot each differenced metal field separately
for column in differenced_df_with_date.columns[1:]:
    plt.figure(figsize=(10, 4))
    plt.plot(differenced_df_with_date.iloc[:, 0], differenced_df_with_date[column])
    plt.title(f'Differenced {column} Prices')
    plt.xlabel('Date')
    plt.ylabel('Differenced Price')
    plt.tight_layout()
    plt.show()

###Lag Selection through ACF and PACF
# Plot ACF and PACF for each metal price
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

for column in differenced_df_with_date.columns[1:]:
    plt.figure(figsize=(10, 4))
    
    # ACF plot
    plt.subplot(1, 2, 1)
    plot_acf(differenced_df_with_date[column].dropna(), ax=plt.gca(), lags=40)
    plt.title(f'Autocorrelation Function ({column})')

    # PACF plot
    plt.subplot(1, 2, 2)
    plot_pacf(differenced_df_with_date[column].dropna(), ax=plt.gca(), lags=40)
    plt.title(f'Partial Autocorrelation Function ({column})')

    plt.tight_layout()
    plt.show()

### AR models
from statsmodels.tsa.arima.model import ARIMA   
# range of AR orders
ar_orders = [1, 2, 3, 4, 5]

# Fit AR models for each metal price field for different AR orders
for column in differenced_df_with_date.columns[1:]:
    print(f"Metal: {column}")
    for ar_order in ar_orders:
        try:
            model = ARIMA(differenced_df_with_date[column], order=(ar_order, 0, 0))  # AR order, no differencing, no MA
            results = model.fit()
            print(f"AR({ar_order}) - AIC: {results.aic}, BIC: {results.bic}")
        except:
            print(f"AR({ar_order}) - Model could not converge.")


##Best AR order for each metal
ar_orders = {
    'Gold': 4,
    'Silver': 2,
    'Platinum': 1,
    'Palladium': 1
    # Add more metals and their corresponding MA orders as needed
}

##Ploting the best AR order for each metal price field
model = ARIMA(differenced_df_with_date['pall'], order=(1, 1, 0))  
results = model.fit()

# Get the fitted values
fitted_values = results.fittedvalues

plt.figure(figsize=(10, 4))
plt.plot(differenced_df_with_date['pall'], label='Actual', color='blue')
plt.plot(fitted_values, label='Fitted', color='orange')
plt.title("Palladium - AR(2) Model")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()


# Define the range of MA orders
ma_orders = [1, 2, 3, 4, 5]

# Fit MA models for each metal price field for different MA orders
for column in differenced_df_with_date.columns[1:]:
    print(f"Metal: {column}")
    for ma_order in ma_orders:
        try:
            model = ARIMA(differenced_df_with_date[column], order=(0, 0, ma_order))  # No AR, no differencing, MA order
            results = model.fit()
            print(f"MA({ma_order}) - AIC: {results.aic}, BIC: {results.bic}")
        except:
            print(f"MA({ma_order}) - Model could not converge.")


# Best MA order for each metal
ma_orders = {
    'Gold': 5,
    'Silver': 4,
    'Platinum': 1,
    'Palladium': 1
    # Add more metals and their corresponding MA orders as needed
}

##Ploting the best MA order for each metal price field
model = ARIMA(differenced_df_with_date['pall'], order=(0, 1, 10))  # No AR, first-order differencing, MA order
results = model.fit()

# Get the fitted values
fitted_values = results.fittedvalues

plt.figure(figsize=(10, 4))
plt.plot(differenced_df_with_date['pall'], label='Actual', color='blue')
plt.plot(fitted_values, label='Fitted', color='orange')
plt.title("Palladium - MA(10) Model")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()


### Lag selection through AIC, BIC

    
# p and q values to iterate over
p_values = range(0, 5)  
q_values = range(0, 5)  

best_aic = float('inf')
best_bic = float('inf')
best_p = 0
best_q = 0

# Iterate over different p and q values to find the best model based on AIC and BIC
for p in p_values:
    for q in q_values:
        for column in differenced_df_with_date.columns[1:]:
            try:
                model = ARIMA(differenced_df_with_date[column], order=(p, 0, q))
                results = model.fit()
                aic = results.aic
                bic = results.bic
                
                if aic < best_aic:
                    best_aic = aic
                    best_p = p
                    best_q = q
                
                if bic < best_bic:
                    best_bic = bic
                    best_p_bic = p
                    best_q_bic = q
                    
            except:
                continue

print(f"Best AIC: p = {best_p}, q = {best_q}, AIC = {best_aic}")
print(f"Best BIC: p = {best_p_bic}, q = {best_q_bic}, BIC = {best_bic}")