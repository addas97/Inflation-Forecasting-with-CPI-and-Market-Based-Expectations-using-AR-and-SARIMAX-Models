# Inflation-Forecasting-with-CPI-and-Market-Based-Expectations-using-AR-and-SARIMAX-Models
Forecasts U.S. inflation using CPI data with linear regression, autoregression (AR), and SARIMAX models. Combines trend and shock decomposition with market-based BER expectations as exogenous inputs. Optimizes ARIMA hyper-parameters and evaluates model performance using RMSE and MAPE.

📊 Inflation Forecasting with CPI and Market-Based Expectations using AR and SARIMAX Models
This project, developed for MIT’s IDS.147[J] Statistical Machine Learning and Data Science course, implements an end-to-end time series modeling pipeline to forecast U.S. inflation using Consumer Price Index (CPI) data and market-based inflation expectations from 10-Year Breakeven Inflation Rate (BER).

🔍 Project Overview
The goal is to build interpretable forecasting models that leverage both trend-based and autoregressive components, enhanced by exogenous inputs. Key modeling steps include:

🧹 Data Preparation
CPI data is cleaned, de-duplicated, and aggregated at a monthly frequency.
BER data is averaged monthly and merged with CPI data to align timeframes.
Inflation is calculated via both percentage change and log differences.

📈 Trend + Residual Modeling (CPI)
Fits a linear regression model to capture long-term CPI inflation trends.
Extracts residuals to model short-term shocks using AutoRegressive (AR) models.
Uses ACF/PACF and RMSE plots to tune AR model order.

🔁 Autoregressive Forecasting
Forecasts future residuals using an AR(1) model.
Combines with trend to reconstruct full inflation forecasts: IR(t) = T(t) + R(t)
Compares predictions to test data and evaluates via RMSE and MAPE.

🌐 SARIMAX with Exogenous Regressors
Uses BER-based inflation as an exogenous input.
Fits SARIMAX models (ARIMA with exogenous variable) to CPI inflation.
Cross-correlation analysis determines if BER leads CPI, validating its use in prediction.

🧪 Hyperparameter Optimization
Evaluates AR, MA, and ARMA combinations (1 ≤ p, q ≤ 10) using:
RMSE on test forecasts
AIC and BIC on training data
Identifies best-fitting (p, d, q) order for SARIMAX.

📉 Final Output
Model summary and optimal order printed.
Visual forecast comparison vs. validation set.
Final model achieves improved accuracy by blending:
Linear trend
AR-modeled shocks
BER as external regressor

⚙️ Tools & Libraries
Python (NumPy, Pandas, Matplotlib, scikit-learn)
Statsmodels (AutoReg, SARIMAX, ACF/PACF analysis)
