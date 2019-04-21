## Design of a ML-based Stock Trader by using Classification Learner

Developed a ML-based trader to optimize trading actions for AAPL shares by training a learner of random forests. Generated training data by using technical features such as simple moving average (SMA), volatility and Bollinger band (BB). Implemented a market simulator that accepts trading orders and keeps track of a portfolio's value over time and then assesses the performance (profit) of the portfolio.


#### Learner Implementation from Scratch

* A Decision Tree-based learner and Bagging are implemented from scratch. See **RTLearner.py** and **BagLearner.py**

#### Market Simulator Implemented

* Implemented a market simulator that accepts trading orders and keeps track of a portfolio's value over time and then assesses the performance of that portfolio. See **marketsim_partX.py**.

#### Manual Rule-Based Trader

* Developed a rule-based method to generate orders book. The rule based method is developed by:
```
Case I (BUY): SMA < 0.95 and bbp < 0
Case II (SELL): SMA > 1.05 and bbp > 1
Case III (HOLD).
```
where SMA is Simple Moving Average, bbp is calcuated by (price - lower band)/(upper band – lower band).

#### Machine Learning Trader

* Developed a rule-based method to generate orders book. The rule based method is developed by:
```
Case I (LONG): ret > YBUY  -->  Y = 1
Case II (SHORT): ret < YSELL --> Y = -1
Case III (HOLD): --> Y = 0
```
where ret reflects the 21-day change in price. ret = (price[t+21]/price[t]) – 1.0

### Running the Tests
Discussion are shown in **Report.pdf**.

### Authors
* **Manqing Mao,** maomanqing@gmail.com

<!-- See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project. -->
