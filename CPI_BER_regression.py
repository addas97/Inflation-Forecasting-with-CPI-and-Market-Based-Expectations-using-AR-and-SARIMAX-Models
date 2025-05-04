# Akash Deep Das
# MIT IDS.147[J] Statistical Machine Learning and Data Science
# CPI / BER Time Series Forecasting Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace import sarimax
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# == CPI Regression == 

# == Load, Clean, and Split Data == 
cpi_data = pd.read_csv('Time_Series_Module4/release_time_series_report_data_nops/CPI.csv')

cpi_data = cpi_data.dropna()
cpi_data['date'] = pd.to_datetime(cpi_data['date'])
cpi_data['Year-Month'] = cpi_data['date'].dt.strftime('%Y-%m')
cpi = cpi_data.drop_duplicates('Year-Month', keep='last').copy().reset_index().drop(['index'],axis=1)

training_data = cpi[cpi['Year-Month'] < '2013-09'].copy()
test_data = cpi[cpi['Year-Month'] >= '2013-09'].copy()

'''# == Plot Raw Data ==
plt.plot(cpi.index, cpi['CPI'])
plt.xlabel('Month')
plt.ylabel('CPI')
plt.title('Monthly CPI vs. Time')
#plt.show()'''

# == Fit Linear Model to CPI Data ==
model = LinearRegression()
model.fit(np.array(training_data.index).reshape(-1, 1), training_data['CPI'])
coef = [model.coef_[0], model.intercept_]
print(f"Linear Regression: {coef[0]} * t + {coef[1]}")
y_train_preds = model.predict(np.array(training_data.index).reshape(-1, 1))
y_test_preds = model.predict(np.array(test_data.index).reshape(-1, 1))

'''# Plot Data - Compare Raw with Fitted
plt.figure(figsize=(6, 4))
plt.plot(training_data.index, training_data['CPI'])
plt.plot(training_data.index, y_train_preds)
plt.xlabel('Month')
plt.ylabel('CPI')
#plt.show()'''

# == Detrend Data ==
residuals_train = training_data['CPI'] - y_train_preds
print(f"Maximum residual value: {np.max(residuals_train)}")

'''# Plot residuals to check for trends and seasonalities
plt.figure(figsize=(6, 4))
plt.plot(training_data.index, residuals_train)
plt.title('Detrended Data - Residual Plot')
plt.xlabel('Months (t)')
plt.ylabel('Residual (Actual - Estimate)')
#plt.show()'''

'''
Reflection:
Based on the plot of the residuals, we do not observe any obvious trend or seasonality in the data.
Thus, we will proceed with a linear model and fit an Autoregression model to the data.
'''

# == Autoregressive Model Fit ==

'''# Plot ACF and PACF to find order p
plot_acf(residuals_train)
#plt.show()
plot_pacf(residuals_train)
#plt.show()
'''

'''
Reflection:
Based on the ACF and PACF plot, the highest lag where the plot extends beyond the statistically significant 
boundary is at p = 2. This should be sufficient to fit the data. 
However, we can build further conviction by calculating the RMSE of the fit.
'''

# Calculate RMSE
def rmse(y_test, y_preds):
    return np.sqrt(mean_squared_error(np.asarray(y_test).flatten(), np.asarray(y_preds).flatten()))

rmse_train = []
for i in range(1, 8):
    model = AutoReg(residuals_train, lags= i, trend = 'n')
    model_fit = model.fit()
    predictions = model_fit.predict()
    rmse_train.append(rmse(residuals_train[i:], predictions[i:]))

'''# Plot RMSE
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, 8), rmse_train)
plt.xlabel('Lag Order')
plt.ylabel('Residuals')
#plt.show()
'''

# Use p = 2 lag order to find model coefficients
model = AutoReg(residuals_train, lags = 2)
model_fitted = model.fit()
coef = model_fitted.params

# == Fit Linear Model to CPI Inflation Data ==
# Calculate Inflation Rate via Percent Change or Log Difference
cpi['IR_pct'] = 0 # OR : cpi['IR'] = cpi['CPI'].pct_change()
for i in range(len(cpi['CPI'])):
    if i > 0:
       cpi['IR_pct'][i] = (cpi['CPI'][i] - cpi['CPI'][i-1]) / cpi['CPI'][i-1]  

cpi['IR_log'] = 0
for i in range(len(cpi['CPI'])):
    if i > 0:
        cpi['IR_log'][i] = (np.log(cpi['CPI'][i]) - np.log(cpi['CPI'][i-1]))

print(f"Inflation Rate on Feb 2013 (via pct change): {cpi.loc[cpi['Year-Month'] == '2013-02', 'IR_pct'].values[0] * 100}%")
print(f"Inflation Rate on Feb 2013 (via log change): {cpi.loc[cpi['Year-Month'] == '2013-02', 'IR_log'].values[0] * 100}%")

cpi['IR_pct'] = cpi['IR_pct'] * 100
cpi['IR_log'] = cpi['IR_log'] * 100

'''plt.figure(figsize=(6, 4))
plt.plot(cpi.index, cpi['IR_pct'])
plt.title('Monthly Inflation Rate via % Change')
plt.xlabel('Month (t)')
plt.ylabel('Inflation Rate')
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(cpi.index, cpi['IR_log'])
plt.title('Monthly Inflation Rate via Log Change')
plt.xlabel('Month (t)')
plt.ylabel('Inflation Rate')
plt.show()'''

# Split Data Into Training / Test Samples
training_data_CPI = cpi[cpi['Year-Month'] < '2013-09'].copy() # Globals - to be used later
testing_data_CPI = cpi[cpi['Year-Month'] >= '2013-09'].copy() # Globals - to be used later

training_len = cpi[cpi['Year-Month'] == '2013-09'].index[0]
x_train = cpi.index[:training_len]
y_train = cpi['IR_pct'][:training_len]
x_test = cpi.index[training_len:]
y_test = cpi['IR_pct'][training_len:]

# Fit Linear Model to Data
lin_mod_ir = LinearRegression()
lin_mod_ir.fit(np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))
y_train_preds_IR = lin_mod_ir.predict(np.array(x_train).reshape(-1, 1)) # T^(t) - training
y_test_preds_IR = lin_mod_ir.predict(np.array((x_test)).reshape(-1, 1)) # T^(t) - testing

# Linear Model Coefficient
a = lin_mod_ir.coef_[0][0]
b = lin_mod_ir.intercept_[0]

# Detrend Data -- Isolate R(t) term
residuals_train_IR = np.array(cpi['IR_pct'][:training_len]).reshape(-1, 1) - y_train_preds_IR # R(t) = IR(t) - T(t)
residuals_test_IR = np.array(cpi['IR_pct'][training_len:]).reshape(-1, 1) - y_test_preds_IR # R(t) = IR(t) - T(t)

'''plt.figure(figsize=(6, 4))
plt.plot(cpi.index[:training_len], residuals_train_IR)
plt.title('Detrended CPI Data')
plt.xlabel('Month (t)')
plt.ylabel('IR(t) = T(t) + R(t) - T(t)')
plt.show()'''

'''
Reflection:
Because the residuals (R(t)) may still have autocorrelation and predictive structure - we can use the past 
residual values to predict future values and add back the T^(t) component to get the full IR^(t) prediction.
'''

# == Fit AR model == 
# Plot ACF/PACF to find order
#plot_acf(residuals_train_IR)
#plt.show()
#plot_pacf(residuals_train_IR)
#plt.show()

# Build model
residuals_all = np.concatenate((residuals_train_IR.flatten(), residuals_test_IR.flatten()))  # All R_hat(t) values in equation: IR(t) = T(t) + R(t) from linear model

def AR(order):
    predictions = []
    for t in range(len(residuals_train_IR), len(residuals_all)): # Roll forward from the end of training set to the end of the test set
        model = AutoReg(residuals_all[:t], lags=order, trend='n') # Use our R^(t) from linear regression to fit AR(order) model and find new R^(t) values up to t - 1 (since splicing rules)
        fitted_model = model.fit()
        pred = fitted_model.predict(start = t, end = t) # end is inclusive -- pred is a 1-element array, pred[0] extracts the scalar
                                                        # Forecasted R^(t) values from AR model - add these to T(t) term to get better IR(t) predictions
        predictions.append(pred.flatten()) # Store forecasted R^(t) values from AR model

    return predictions

order = 1
predictions = AR(order = order)

y_test_preds_IR = y_test_preds_IR.flatten()
predictions = np.array(predictions).flatten() # Final R^(t) component - modeled via the AR process
predicted_ir_test = y_test_preds_IR + predictions # Full model: IR^(t) = T^(t) + R^(t)

'''plt.figure(figsize=(10, 5))
plt.plot(x_test, y_test, label='Validation', linewidth=2, color = 'blue')
plt.plot(x_test, predicted_ir_test, label='Predicted', linestyle='-.', linewidth=2, color = 'red')
plt.xlabel("t (month)")
plt.ylabel("IRₜ")
plt.title("Forecasting Inflation Rates: Validation Set")
plt.legend()
plt.show()
'''

print(f"Final Inflation Rate Forecasting Model IR(t) = ({a:.5f}) * t + ({b:.5f}) + AR({order})")

# RSME test
def compute_rmse_for_lags(max_lag):
    rmse_by_lag = []
    for lag in range(1, max_lag + 1):
        predictions = []
        actuals = []
        for t in range(len(residuals_train_IR), len(residuals_all)):
            model = AutoReg(residuals_all[:t], lags=lag, trend='n')
            fitted_model = model.fit()
            pred = fitted_model.predict(start=t, end=t)[0]  # scalar
            predictions.append(pred)
            actuals.append(residuals_all[t])  # true value at time t
        rmse_ = rmse(actuals, predictions)
        rmse_by_lag.append(rmse_)
    
    return rmse_by_lag

rmse_by_lag = compute_rmse_for_lags(max_lag = 11)

def AIC_BIC_for_lags(max_lag):
    aic_values = []
    bic_values = []

    for lag in range(1, 11):
        model = AutoReg(residuals_train_IR, lags=lag, trend='n')
        fit = model.fit()
        aic_values.append(fit.aic)
        bic_values.append(fit.bic)

    return aic_values, bic_values

aic_values, bic_values = AIC_BIC_for_lags(max_lag = 11)

'''plt.figure(figsize=(10, 5))
plt.plot(range(len(rmse_by_lag)), rmse_by_lag, linewidth=2, color = 'red')
plt.xlabel("AR Lag (p)")
plt.ylabel("RMSE")
plt.title("RMSE for AR Model at Different Lags")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(len(aic_values)), aic_values, linewidth=2, color = 'blue')
plt.plot(range(len(bic_values)), bic_values, linewidth=2, color = 'red')
plt.xlabel("k")
plt.ylabel("Value")
plt.title("AIC/BIC for AR Model at Different Lags")
plt.show()'''

# == BER Regression == 

# == Load, Clean, and Split Data == 
ber_data = pd.read_csv('Time_Series_Module4/release_time_series_report_data_nops/T10YIE.csv')
ber_data['DATE'] = pd.to_datetime(ber_data['DATE'])
ber_data['Year-Month'] = ber_data['DATE'].dt.strftime('%Y-%m')
ber_data = ber_data[ber_data['Year-Month'] >= '2008-07']
ber_data = ber_data.dropna()

ber_monthly = ber_data.groupby('Year-Month')['T10YIE'].mean().reset_index()
ber_monthly.columns = ['Year-Month', 'BER']
cpi_ber = pd.merge(cpi, ber_monthly, on='Year-Month', how='inner')  # Use inner to keep only matching months
ber_monthly = ber_data.groupby('Year-Month')['T10YIE'].mean().reset_index()
ber_monthly.columns = ['Year-Month', 'BER']

final_ber = pd.merge(cpi[['Year-Month']], ber_monthly, on='Year-Month', how='inner')
final_ber['IR_BER'] = ((final_ber['BER'].values / 100 + 1) ** (1 / 12) - 1) * 100

training_data_BER = final_ber[final_ber['Year-Month'] < '2013-09'].copy()
test_data_BER = final_ber[final_ber['Year-Month'] >= '2013-09'].copy()

'''plt.figure(figsize=(10, 5))
plt.plot(x_test, y_test, label='CPI', linewidth=2, color = 'blue')
plt.plot(test_data_BER.index, test_data_BER['IR_BER'], label='BER', linewidth=2, color = 'orange')
plt.plot(x_test, predicted_ir_test, label='Predicted', linestyle='-.', linewidth=2, color = 'red')
plt.xlabel("t (month)")
plt.ylabel("IRₜ")
plt.title("Forecasting Inflation Rates - Validation Set with CPI and BER Data")
plt.legend()
plt.show()'''

# == Plot Cross-Correlation between CPI and BER Inflation Rates ==
# We use the BER data as an external regressor to improve our predictions. The CCF will help us find the appropriate lag between the BER and the CPI.

'''
Reflection:
Lag    Interpretation
0	   Do BER and CPI move together?
+1	   Does BER at time t affect CPI at time t + 1?
+2	   Does BER at time t affect CPI t + 2 months later?
-1	   Does CPI at time t BER at time t + 1?

Summary:
Positive lag → BER leads CPI
→ Changes in BER precede changes in CPI
→ BER might help predict future CPI

Negative lag → BER lags CPI
→ Changes in CPI precede changes in BER
→ CPI might influence or reflect changes in BER

Lag = 0 → BER and CPI move together
→ They may be responding to the same macroeconomic forces simultaneously
''' 

'''plt.figure(figsize = (10, 6))
plt.xcorr(training_data_BER.IR_BER[1:], y_train[1:], maxlags = 25)
plt.title('Cross-Correlation between CPI and BER Inflation Rates')
plt.show()'''

# == Fit New AR Model to CPI Data w/ BER Data == 

# Create Training / Test Set
exogenous_variable_BER_train = training_data_BER.copy().drop(['BER', 'Year-Month'], axis = 1)
endogenous_variable_CPI_train = training_data_CPI.copy().drop(['date', 'CPI', 'Year-Month', 'IR_log'], axis = 1)
exogenous_variable_BER_test = test_data_BER.copy().drop(['BER', 'Year-Month'], axis = 1)
endogenous_variable_CPI_test = testing_data_CPI.copy().drop(['date', 'CPI', 'Year-Month', 'IR_log'], axis = 1)

# Train Model
def fit_combined_model(p, d, q):
    combined_model = sm.tsa.statespace.SARIMAX(endog = endogenous_variable_CPI_train, exog = exogenous_variable_BER_train, order = (p, d, q), coerce_errors = True).fit(disp = False) # Order: (AR param, differencing param, MA param), use because SARIMAX supports exogenous regressors
    combined_model_preds = combined_model.predict(start = 0, end = len(exogenous_variable_BER_train.index) - 1, exog = exogenous_variable_BER_train) # Training preds
    combined_model_coef = combined_model.params
    return combined_model, combined_model_preds, combined_model_coef

# Make Test Set Predictions

# Start with AR(1) model: IR_CPI(t)= ϕ1 * IR_CPI(t−1)+β⋅IR_BER(t)+ϵ(t) --> CPI this month relies on CPI last month and BER this month
p_order = 1
AR1_combined_model, AR1_combined_model_preds_train, AR1_combined_model_coef = fit_combined_model(p = p_order, d = 0, q = 0)
AR1_combined_model_preds_test = AR1_combined_model.predict(start = len(training_data_CPI), end = len(training_data_CPI) + len(testing_data_CPI) - 1, exog = exogenous_variable_BER_test)
print(AR1_combined_model.summary())

# Predict On Test Set
def pred(endogenous_variable_CPI_train, exogenous_variable_BER_test, endogenous_variable_CPI_test, p_order, coef):
    past_obs = list(endogenous_variable_CPI_train['IR_pct'].iloc[-p_order:].values)
    test_preds = []

    for t in range(len(endogenous_variable_CPI_test)):
        pred = 0

        # AR terms
        for i in range(1, p_order + 1):
            coef_name = f'ar.L{i}'
            pred += coef[coef_name] * past_obs[-i]

        # Exogenous input (IR_BER)
        pred += coef['IR_BER'] * exogenous_variable_BER_test.iloc[t].values[0]

        test_preds.append(float(pred))
        past_obs.append(float(endogenous_variable_CPI_test.iloc[t]))

    return test_preds

test_preds = np.asarray(pred(endogenous_variable_CPI_train, exogenous_variable_BER_test, endogenous_variable_CPI_test, p_order, AR1_combined_model_coef)).flatten()

plt.figure(figsize=(10, 6))
plt.plot(endogenous_variable_CPI_test.index, endogenous_variable_CPI_test, label='Validation', linewidth=2, color = 'blue')
plt.plot(endogenous_variable_CPI_test.index, test_preds, label='Predicted', linestyle='-.', linewidth=2, color = 'red')
plt.xlabel("t (month)")
plt.ylabel("IRₜ")
plt.title("Forecasting Inflation Rates using External Regressors: Validation Set")
plt.legend()
plt.show()

print("RMSE:", mean_squared_error(np.asarray(endogenous_variable_CPI_test).flatten(), test_preds) ** 0.5)
print("MAPE:", mean_absolute_percentage_error(np.asarray(endogenous_variable_CPI_test).flatten(), test_preds)) 

# == Hyperparameter Testing (AR parameters) ==
best_rmse_AR = float('inf')
ideal_p = 0.0
for ar in range(1, 11):
    combined_model_AR_only, _, combined_model_coef_AR_only = fit_combined_model(ar, 0, 0)
    test_preds_AR = pred(endogenous_variable_CPI_train, exogenous_variable_BER_test, endogenous_variable_CPI_test, p_order, combined_model_coef_AR_only)
    score = rmse(endogenous_variable_CPI_test, test_preds_AR)

    if score < best_rmse_AR:
        best_rmse_AR = score
        ideal_p = ar

print(f"Best RMSE: {best_rmse_AR:.4f} with order = {ideal_p}")

'''# == Hyperparameter Testing (MA parameters) ==
def pred_MA(endogenous_variable_CPI_train, exogenous_variable_BER_test, endogenous_variable_CPI_test, q_order, coef):
    past_obs = list(endogenous_variable_CPI_train['IR_pct'].iloc[-p_order:].values)
    test_preds = []

    for t in range(len(endogenous_variable_CPI_test)):
        pred = 0

        # AR terms
        for i in range(1, p_order + 1):
            coef_name = f'ar.L{i}'
            pred += coef[coef_name] * past_obs[-i]

        # Exogenous input (IR_BER)
        pred += coef['IR_BER'] * exogenous_variable_BER_test.iloc[t].values[0]

        test_preds.append(float(pred))
        past_obs.append(float(endogenous_variable_CPI_test.iloc[t]))

    return test_preds

best_rmse_AR = float('inf')
ideal_q = 0.0

for ma in range(1, 11):
    combined_model_AR_only, _, combined_model_coef_AR_only = fit_combined_model(ar, 0, 0)
    test_preds_AR = pred(endogenous_variable_CPI_train, exogenous_variable_BER_test, endogenous_variable_CPI_test, p_order, combined_model_coef_AR_only)
    score = rmse(endogenous_variable_CPI_test, test_preds_AR)

    if score < best_rmse_AR:
        best_rmse_AR = score
        ideal_p = ar
'''

# == Hyperparameter Testing (AR and MA parameters) ==
def pred_p_q_order(endogenous_variable_CPI_train, exogenous_variable_BER_test, endogenous_variable_CPI_test, p_order, q_order, coef):
    past_obs = list(endogenous_variable_CPI_train['IR_pct'].iloc[-p_order:].values)
    past_errors = [0] * q_order
    test_preds = []

    for t in range(len(endogenous_variable_CPI_test)):
        pred = 0
        
        # AR term preds
        for i in range(1, p_order + 1):
            coef_name = f'ar.L{i}'
            pred += coef[coef_name] * past_obs[-i]

        # MA term preds
        for j in range(1, q_order + 1):
            pred += coef.get(f'ma.L{j}', 0) * past_errors[-j]
        
        # Exogenous variable preds
        pred += coef['IR_BER'] * exogenous_variable_BER_test.iloc[t].values[0]
        test_preds.append(pred)

        # Update past observations
        actual = float(exogenous_variable_BER_test.iloc[t])
        past_obs.append(actual)
        error = actual - pred
        past_errors.append(error)

    return test_preds

best_rmse = float('inf')
best_order = None

for ar in range(1, 11):
    rmse_ = []
    for ma in range(1, 11):
        try:
            combined_model, _, combined_model_coef = fit_combined_model(p = ar, d = 0, q = ma)
            test_preds = pred_p_q_order(endogenous_variable_CPI_train = endogenous_variable_CPI_train, exogenous_variable_BER_test = exogenous_variable_BER_test, endogenous_variable_CPI_test = endogenous_variable_CPI_test, p_order = ar, q_order = ma, coef = combined_model_coef)
            score = rmse(np.asarray(endogenous_variable_CPI_test).flatten(), test_preds)

            if score < best_rmse:
                best_rmse = score
                best_order = (ar, 0, ma)
        except:
            pass

print(f"Best RMSE: {best_rmse:.4f} with order = {best_order}")