# Power Outage Cause Predictions 
Predicting the cause of power outages in the continental U.S.

## Framing the Problem 

In my [last project using this dataset](https://matteoperona.github.io/usa-power-outage-analysis/) I looked into the relationship between power outages per capita and instances of intentional attack relative to other outage causes. I found that areas with high outage per capita were much more likely to have a relatively high proportion of outages caused by intentional attacks. This time, I will explore the same vein of interest backwards: training some models to predict the cause category (**CAUSE.CATEGORY** in the dataset) of a given power outage. I chose cause category because it broadly categorizes the cause of a given power outage into seven categories ('severe weather', 'intentional attack', 'system operability disruption', 'equipment failure', 'public appeal', 'fuel supply emergency', and 'islanding') which lend themselves well to a **multiclass classification**. This model will, hopefully, give us a finer understanding of outage causes, and help us to understand which features are most relevant when assessing a given region's propensity toward certain types of outage. Equipped with this information, municipalities might better prepare for future outages.

For this **classification** problem all columns in the dataset should be fair game except for the CAUSE.CATEGORY.DETAIL and HURRICANE.NAME as they would both be unknown if we did not know the cause of the outage. The rest of our dataset contains information about datetime, location, climate, extent, regional electricity consumption, regional economic characteristics, and regional land-use characteristics, all of which could be known before the cause. 

To assess the performance of the model we will be using accuracy, and F1 score.

## Baseline Model

To start we'll create a baseline model using:
- U.S._STATE (nominal variable containing the US State in which the outage occurred)
- OUTAGE.DURATION (quantitative variable containing the outage duration in minutes)

To predict:
- CAUSE.CATEGORY (nominal variable describing the cause of a given power outage)

> We are removing Alaska from our data because there are not enough observations to make it viable for our model. <br>
> We are not using any ordinal data for this model.

Let's grab our baseline data from our outage dataframe:

```py
# Separate relevant data 
baseline_data = outage[['U.S._STATE', 'OUTAGE.DURATION', 'CAUSE.CATEGORY', 'CUSTOMERS.AFFECTED']]

# Remove Alaska from the data because there arent enough observations to include it in our model
baseline_data = baseline_data[baseline_data['U.S._STATE'] != 'Alaska']
```

Here's the baseline_data's head:

| U.S._STATE   |   OUTAGE.DURATION | CAUSE.CATEGORY     |   CUSTOMERS.AFFECTED |
|:-------------|------------------:|:-------------------|---------------------:|
| Minnesota    |              3060 | severe weather     |                70000 |
| Minnesota    |                 1 | intentional attack |                  nan |
| Minnesota    |              3000 | severe weather     |                70000 |
| Minnesota    |              2550 | severe weather     |                68200 |
| Minnesota    |              1740 | severe weather     |               250000 |

### Step One: Train Test Split

Our first step is to separate our data into training and testing data. We separate variables we are predicting from the rest of our data then sample our data at random, reserving 80% as training data to fit our model and 20% to test our model on unseen data.

```py
df = baseline_data

# Define X and y 
X = df.drop(columns=['CAUSE.CATEGORY'])
y = df['CAUSE.CATEGORY']

# Split into training and testing data (stratifying by U.S._State)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=X['U.S._STATE'])
```

 One important thing to note at this stage is that we had to stratify our data when splitting into training and testing sets using the U.S._STATE variable. This makes sure that both training and testing data have relatively equal proportions of observations from each state. Without this step, we could run into problems where not all states represented in the testing set were included -- and one-hot encoded -- with the training set.

### Step Two: Preprocessing and Pipeline

Next, we want to create our column transformer to preprocess our data and our pipeline to run the preprocesses and fit our model. 

We are only using two transformers:
- A KNNImputer to handle missing data in the OUTAGE.DURATION column
- A OneHotEncoder to encode the nominal data in the U.S._STATE column

I've gone for a random forest classifier here because it's [less influenced by outliers](https://stats.stackexchange.com/questions/187200/how-are-random-forests-not-sensitive-to-outliers) than other algorithms and can [implicitly handle multicollinearity](https://stats.stackexchange.com/questions/141619/wont-highly-correlated-variables-in-random-forest-distort-accuracy-and-feature). 

```py
# Stage all preprocessing steps
preproc = ColumnTransformer(
    transformers=[
        ('impute_duration', KNNImputer(), ['OUTAGE.DURATION', 'CUSTOMERS.AFFECTED']),
        ('one_hot', OneHotEncoder(), ['U.S._STATE']),
    ],
    remainder='passthrough'
)

# Create final pipeline
pl = Pipeline([
    ('preprocess', preproc), 
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6))
])
```

### Step Three: Fit and Model Eval 

Finally, we can fit our model with out training data and evaluate how "good" our predictions are.

```py
# Fit model to training data
pl.fit(X_train, y_train)

# Make predictions 
y_pred = pl.predict(X_test)

# Calculate accuracy and f1 score 
acc = (y_pred == y_test).mean()
f1 = f1_score(y_pred=y_pred, y_true=y_test, labels=y.unique(), average='weighted')
```

Here we are using using a weighted average on each partial f1 score to come up with out final value. This gives us the weighted average of the the harmonic means of precision and recall for each cause category: a value between 0 (worst) and 1 (best) telling us how well our model performed. 

#### Baseline Model Performance

Accuracy: 0.69

F1: .61

The baseline model seems to have given lukewarm results. While the accuracy is nearly 70% the F1 score is about .1 less, which tells us that recall (True Positive / All Actual Positive) is not stacking up. To give a better understanding here is a confusion matrix for this model. 

<img src="./assets/output.png"
     alt="Markdown Monster icon"/>

Pay attention to the diagonal, this is where the count of correct predictions lies. We can see that intentional attack, and severe weather are the only two categories which are being predicted with any accuracy. This makes sense because we can see that they are, by far, the most well represented categories in our dataset. If we did not take a weighted average when calculating our overall f1 score, it would be much lower since the model is really only able to predict those two categories. 

So is this model good? I would conclude that it is not good because it can only predict two categories with any accuracy, and, even then, the model's recall is not very good. 

## Final Model 

The final model will consist of the following variables for the following reasons:
1. U.S._STATE (string; the name of the U.S. state in which the outage occurred)
   - Lends geographical context to the data; certain outage causes are probably more common in select states
2. ANOMALY.LEVEL (float64; This represents the oceanic El Niño/La Niña (ONI) index referring to the cold and warm episodes by season)
   - Gives perspective on climate during that power outage. If El Niño is in full swing, it may be far more likely to see outages caused by extreme weather.
3. OUTAGE.DURATION (float64; The duration of the power outage in minutes)
   - Adds temporal context. Longer outages may be linked with select causes.
5. DEMAND.LOSS.MW (float64; Amount of peak demand lost during an outage event (in Megawatt))
   - Gives context to the extent of the power outage from the grid's perspective. Knowing the extent to which the outage impacted the grid could be very indicative of the type of cause of the outage. Certain causes probably lead to much more drastic losses.
6. CUSTOMERS.AFFECTED (float64; Number of customers affected by the power outage event)
   - Gives context to the extent of the power outage from a population perspective which could indicate. The number of people affected, like the amount of demand lost, could indicate higher likelihood for certain causes.
7. AREAPCT_URBAN (float64; Percentage of the land area of the U.S. state represented by the land area of the urban areas)
   - Adds context about land use. More urban land could indicate higher risk for intentional attacks, or, perhaps, less urban land would indicate a higher propensity for wind storms.
8. PCT_LAND (float64; Percentage of land area in the U.S. state as compared to the overall land area in the continental U.S.)
   - Describes the size of the state in which the outage occurred relative to other U.S. states. The smaller the state is the more titrated its list of climate related outages might be.
9. PCT_WATER_TOT (float64; Percentage of water area in the U.S. state as compared to the overall water area in the continental U.S.)
   - Describes the percentage of all water in the U.S. contained within the outage's state. This information could imply certain climate conditions which could indicate a higher propensity for certain types of power outage.
10. POPULATION (int64; Population in the U.S. state in a year)
   - This adds context to CUSTOMERS.AFFECTED and PCT_LAND. The proportion of people in the state affected by the outage as well as the proportion of people in the state relative to its size could both be important indicators of a given cause.

>
> OUTAGE.DURATION was removed from the final model since OUTAGE.START and OUTAGE.RESTORATION implicitly give the duration.
>

