# Power Outage Cause Predictions 
Predicting the cause of power outages in the continental U.S.

Report the response variable (i.e. the variable you are predicting) and why you chose it, the metric you are using to evaluate your model and why you chose it over other suitable metrics (e.g. accuracy vs. F1-score). 

Note: Make sure to justify what information you would know at the “time of prediction” and to only train your model using those features. For instance, if we wanted to predict your final exam grade, we couldn’t use your Project 5 grade, because Project 5 is only due after the final exam!

## Framing the Problem 

In my [last project using this dataset](https://matteoperona.github.io/usa-power-outage-analysis/) I looked into the relationship between power outages per capita and instances of intentional attack relative to other outage causes. I found that areas with high outage per capita were much more likely to have a relatively high proportion of outages caused by intentional attacks. This time, I will explore the same vein of interest backwards: training some models to predict the cause category (**CAUSE.CATEGORY** in the dataset) of a given power outage. I chose cause category because it broadly categorizes the cause of a given power outage into seven categories: 'severe weather', 'intentional attack', 'system operability disruption', 'equipment failure', 'public appeal', 'fuel supply emergency', and 'islanding'. This **multiclass classification** problem will, hopefully, give us a finer understanding of outage causes, and help us to understand which features are most relevant when assessing a given region's propensity toward certain types of outage. Equipped with this information, municipalities might better prepare for future outages.

For this **classification** problem all columns in the dataset should be fair game except for the CAUSE.CATEGORY.DETAIL and HURRICANE.NAME as they would both be unknown if we did not know the cause of the outage. The rest of our dataset contains information about datetime, location, climate, extent, regional electricity consumption, regional economic characteristics, and regional land-use characteristics, all of which could be known before the cause. 

To assess the performance of the model we will be using RMSE, accuracy, and F1 score.


