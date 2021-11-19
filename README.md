# jobathon_nov21
data preparation notes:

1. since we need to predict whether an employee resigns in a future 6 month window, we need to check against each
record in the data (every month) whether the employee resigns
2. this is done by comparing the difference in duration between the snapshot date column (yyy-mm) and resignation date column - the difference should be between 0 and 6 months
3. we will need to decide a window in which we will observe an employees history, and use that data to predict resignation
4. the length of this lookback window can be determined either via EDA, or by including it as a hyperparameter.

raw data exploration observations:

1. raw data has no missing values, except for the last working day column
2. high correlation observed (as is expected) between age, salary, and designation
3. high correlation observed (as is expected) between total business value, quarterly rating, and our target column

Given our own experiences with appraisal cycles in general, there seems to be a lot of scope to enhance the model performance by creating custom features. Here are a few feature engineering ideas and notes:

1. when employees dont get promoted despite good performance
2. large number of years since previous promotion
3. salary compared to peers (definition of peers can be wrt city, designation, gender , education etc)
4. change in city / transfer
5. brand new employee
6. tenure
7. demotions
8. measure the appraisal cycle length
9. compare employee performance against best and worst employee in their own band