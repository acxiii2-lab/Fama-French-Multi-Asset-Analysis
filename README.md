# Fama-French Multi Asset Analysis

## Overview

This project applies the Fama-French Three-Factor Model to a portfolio of UK-listed equities.

The objective is to evaluate whether market, size, and value factors explain portfolio returns and to identify the dominant sources of performance.

---

## Assets Included

- Barclays (BARC.L)
- Shell (SHEL.L)
- Vodafone (VOD.L)
- Bluebird Mining Ventures (BLU.L)

---

## Methodology

1. Download historical price data using yFinance
2. Calculate monthly returns
3. Construct equal-weight portfolio
4. Obtain Fama-French factors
5. Run OLS regression
6. Analyse factor exposure and explanatory power

---

## Tools

- Python
- Pandas
- NumPy
- Statsmodels
- Matplotlib
- yFinance

---

## Key Findings

- Market factor explains the majority of portfolio returns.
- Size and Value factors contribute differently across sectors.
- R-squared analysis suggests that factor models explain a significant portion of return variation.

---

## Research Skills Demonstrated

- Financial Data Analysis
- Quantitative Research
- Regression Modelling
- Factor Investing
- Portfolio Analytics

---

## Correlation Heatmap

![Correlation](images/r2_heatmap.png)


---

## Regression Result

![OLS](images/logistic_summary.png)

---

## Research Conclusion

This study examines whether traditional Fama-French factors explain the returns of selected UK equities.

The analysis begins by converting monthly stock prices into monthly returns and aligning them with the Fama-French market (Mkt-RF), size (SMB), value (HML), and risk-free rate (RF) factors. Multiple regression models are then estimated to compare the explanatory power of different predictor sets.

The results suggest that:

Fama-French factors provide meaningful explanatory power for asset returns, but they do not fully explain stock performance on their own.

Including returns from other assets alongside the Fama-French factors generally improves model performance, indicating that co-movement among stocks contains additional information beyond the classical factor model.

Adding a random control variable produces little or no improvement, supporting the view that the observed explanatory power comes from economically meaningful variables rather than random noise.

Logistic regression shows that future return direction can be predicted better than chance for some assets, although prediction accuracy varies across securities and remains limited by market uncertainty.

Overall, the findings indicate that the Fama-French framework remains a useful foundation for explaining equity returns, but combining factor exposures with cross-asset information provides a more comprehensive representation of market behaviour.


---
## Author

AC HSUEH

MSc Finance | CFA Level I Candidate

London, United Kingdom# Fama-French Multi Asset Analysis

This project applies the Fama-French Three Factor Model
to a portfolio of UK listed equities.

Assets analysed:

- Barclays (BARC)
- Shell (SHEL)
- Vodafone (VOD)
- BAE Systems (BA)

Methods:

- Linear Regression
- Logistic Regression
- Factor Attribution
- ROC Analysis
- Confusion Matrix
- R² Comparison

Tools:

- Python
- Pandas
- Scikit-Learn
- Statsmodels
- Matplotlib
