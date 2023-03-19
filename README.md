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
> Note: We are not using any ordinal data for this model.

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

Our first step is to separate our data into training and testing data. We separate variables we are predicting from the rest of our data then sample our data at random, reserving 80% as training data to fit our model and 20% to test our model on unseen data. One important thing to note at this stage is that we had to stratify our data when splitting into training and testing sets using the U.S._STATE variable. This makes sure that both training and testing data have relatively equal proportions of observations from each state. Without this step, we could run into problems where not all states represented in the testing set were included -- and one-hot encoded -- with the training set.

```py
df = baseline_data

# Define X and y 
X = df.drop(columns=['CAUSE.CATEGORY'])
y = df['CAUSE.CATEGORY']

# Split into training and testing data (stratifying by U.S._State)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=X['U.S._STATE'])
```

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

