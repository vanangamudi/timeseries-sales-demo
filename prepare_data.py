## import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.dates import MonthLocator, DateFormatter
from datetime import datetime
 
import os

plt.style.use('ggplot')
matplotlib.use( 'tkagg' )


## reading train and test data
train_all = pd.read_csv('../data/store/train.csv')
test_raw = pd.read_csv('../data/store/test.csv')

## checking unique values in most important groups
train_all[["store_nbr","family","date"]].nunique()

def mk_date(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df["date"].dt.year
    df['month'] = df["date"].dt.month
    #df['week'] = df["date"].dt.week
    df['weekday'] = df["date"].apply(lambda x: x.weekday())
    return df


train_all = mk_date(train_all)
test_raw  = mk_date(test_raw)


def mk_max_promotions(df):
    max_onpromotions = df.groupby("family")['onpromotion'].agg(['max']).reset_index()
    max_onpromotions.columns = ["family", "maxpromoted"]
    max_onpromotions["maxpromoted"] = max_onpromotions["maxpromoted"] + 0.0001
    return max_onpromotions

max_onpromotions = mk_max_promotions(train_all)

train_all = train_all.merge(max_onpromotions, on = "family", how = "left")
train_all["onpromotion_perc"] = train_all['onpromotion'] / train_all['maxpromoted']
test_raw = test_raw.merge(max_onpromotions, on = "family", how = "left")
test_raw["onpromotion_perc"] = test_raw['onpromotion'] / test_raw['maxpromoted']

analysis = train_all[
    (train_all["store_nbr"]==3)
    & (train_all["family"]=="SCHOOL AND OFFICE SUPPLIES")
    & (train_all["year"]== 2017)
]

fig, ax = plt.subplots(figsize=(15, 5))


ax.plot(analysis['date'], analysis['sales'])
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter("%b"))
ax.set_xlabel('Month')
ax.set_ylabel('Sales')
ax.set_title('Daily Sales in Monthly Resolution 2016')
plt.xticks(rotation=45)
plt.show()

is_actual_training = 0
if is_actual_training:
    train_filter_date = "2017-08-16"
else:
    train_filter_date = "2017-08-01"

validation_data = train_all[train_all["date"] >= train_filter_date ]
train_data = train_all[train_all["date"] < train_filter_date ]

print(validation_data)
exit(0)

def calculateSEForGroup( df, group= ['store_nbr', 'family'], identifier="General"):
    
    # define column names 
    mean_col = 'Mean'+identifier
    std_col = 'StdDev'+identifier
    standard_error_col = "SE_"+ identifier
    
    # calculate stats per group
    grouped = df.groupby(group)['sales'].agg(['mean', 'std']).reset_index()

    # Rename the columns  
    grouped = grouped.rename(columns={'mean': mean_col, 'std': std_col})
    df = df.merge(grouped, on= group, how="left")
    
    df[standard_error_col] = (df["sales"] - df[mean_col]) / (df[std_col] + 0.001)  # to prevent zero division error
    df = df.drop([mean_col,std_col], axis=1)
    return df

def applyStandardErrors(df):
    df = calculateSEForGroup(df) #1st
    df = calculateSEForGroup(df, group = ['store_nbr', 'family','year'], identifier ="Yearly" ) #2nd
    df = calculateSEForGroup(df, group = ['store_nbr', 'family','month'], identifier ="Monthly" ) #3rd
    df = calculateSEForGroup(df, group = ['store_nbr', 'family','weekday'], identifier ="Dayofweek" ) #4th
    return df


train_data = applyStandardErrors(train_data)
train_data


plt.hist(train_data["SE_Dayofweek"], bins = 15, range=[-3,3])

day_of_week_filtered = train_data.loc[(train_data["SE_Dayofweek"]< 3) & (train_data["SE_Dayofweek"]> -3),["sales","weekday"]]
#day_of_week_filtered = train_data.loc[(train_data["date"]> '2015-01-01')  ,["sales","weekday"]]
day_of_week_filtered = day_of_week_filtered.groupby("weekday").mean()
day_of_week_filtered = day_of_week_filtered.reset_index()
day_of_week_filtered["Dayofweek_factor"] = 7 * day_of_week_filtered["sales"] / sum(day_of_week_filtered["sales"])
day_of_week_filtered = day_of_week_filtered.drop(["sales"], axis= 1)

train_data = train_data.merge(day_of_week_filtered, on="weekday")# join this to full data later
validation_data = validation_data.merge(day_of_week_filtered, on="weekday")# join this to full data later
test_raw = test_raw.merge(day_of_week_filtered, on="weekday")# join this to full data later

print(day_of_week_filtered)
validation_data

## apparently all the stores has full data (regarding the dates)
stores_min_date = train_data.groupby("store_nbr")['date'].agg(['min','max']).reset_index()
stores_min_date[["min","max"]].value_counts()


## calculate sales volumes of stores
stores_sales = train_data.groupby("store_nbr")['sales'].agg(['sum']).reset_index()
stores_sales["store_factor"] = stores_sales.shape[0] * stores_sales["sum"] / sum(stores_sales["sum"])
stores_sales = stores_sales.drop("sum",axis = 1)
plt.hist(stores_sales["store_factor"], bins= 30)

## calculate sales volumes of stores for the most recent year
stores_sales_recent = train_data[train_data["date"] > '2017-01-01'].groupby("store_nbr")['sales'].agg(['sum']).reset_index()
stores_sales_recent["store_factor_ty"] = stores_sales_recent.shape[0] * stores_sales_recent["sum"] / sum(stores_sales_recent["sum"])
stores_sales_recent = stores_sales_recent.drop("sum",axis = 1)
plt.hist(stores_sales_recent["store_factor_ty"], bins= 30)


max_date = train_data["date"].max()

avg_predictions7 = train_data[train_data["date"]> max_date - pd.Timedelta(days=7)]
# ensure getting 14 days (better have a multiple of 7 days to get rid of weekday effect distortion in this step)
print(str(avg_predictions7["date"].nunique()) + " days")

avg_predictions14 = train_data[train_data["date"]> max_date - pd.Timedelta(days=14)]
# ensure getting 14 days (better have a multiple of 7 days to get rid of weekday effect distortion in this step)
print(str(avg_predictions14["date"].nunique()) + " days")

avg_predictions28 = train_data[train_data["date"]> max_date - pd.Timedelta(days=28)]
# ensure getting 28 days (better have a multiple of 7 days to get rid of weekday effect distortion in this step)
print(str(avg_predictions28["date"].nunique()) + " days")

avg_predictions42 = train_data[train_data["date"]> max_date - pd.Timedelta(days=42)]
# ensure getting 42 days (better have a multiple of 7 days to get rid of weekday effect distortion in this step)
print(str(avg_predictions42["date"].nunique()) + " days")

avg_predictions7 = avg_predictions7.groupby(["store_nbr","family"])["sales"].agg("mean").reset_index()
avg_predictions7.columns = ["store_nbr","family","MA7"]

avg_predictions14 = avg_predictions14.groupby(["store_nbr","family"])["sales"].agg("mean").reset_index()
avg_predictions14.columns = ["store_nbr","family","MA14"]

avg_predictions28 = avg_predictions28.groupby(["store_nbr","family"])["sales"].agg("mean").reset_index()
avg_predictions28.columns = ["store_nbr","family","MA28"]

avg_predictions42 = avg_predictions42.groupby(["store_nbr","family"])["sales"].agg("mean").reset_index()
avg_predictions42.columns = ["store_nbr","family","MA42"]

avg_predictions42


train_data = train_data.merge(avg_predictions7, on=["store_nbr","family"], how = "left")
train_data = train_data.merge(avg_predictions14, on=["store_nbr","family"], how = "left")
train_data = train_data.merge(avg_predictions28, on=["store_nbr","family"], how = "left")
train_data = train_data.merge(avg_predictions42, on=["store_nbr","family"], how = "left")

validation_data_mod = validation_data.merge(avg_predictions7, on=["store_nbr","family"], how = "left")
validation_data_mod = validation_data_mod.merge(avg_predictions14, on=["store_nbr","family"], how = "left")
validation_data_mod = validation_data_mod.merge(avg_predictions28, on=["store_nbr","family"], how = "left")
validation_data_mod = validation_data_mod.merge(avg_predictions42, on=["store_nbr","family"], how = "left")
validation_data_mod


validation_data_mod["pred"] = validation_data_mod["MA14"] * validation_data_mod["Dayofweek_factor"] 
validation_data_mod.loc[ validation_data_mod.family=="SCHOOL AND OFFICE SUPPLIES", "pred"] = validation_data_mod.loc[ validation_data_mod.family=="SCHOOL AND OFFICE SUPPLIES",  "MA7"] * validation_data_mod.loc[ validation_data_mod.family=="SCHOOL AND OFFICE SUPPLIES",  "Dayofweek_factor"] 
validation_data_mod


def rmsle(y_actual, y_predicted):
    log_actual = np.log1p(y_actual)
    log_predicted = np.log1p(y_predicted)
    squared_errors = (log_actual - log_predicted) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmsle = np.sqrt(mean_squared_error)
    return rmsle


def mad(y_actual, y_predicted):
    log_actual = np.log1p(y_actual)
    log_predicted = np.log1p(y_predicted)     
    return np.mean(log_predicted - log_actual)

print(rmsle(validation_data_mod["sales"],validation_data_mod["pred"]))
#validation_data_mod = validation_data_mod.drop("pred", axis = 1)

family_result = validation_data_mod.groupby('family').apply(lambda group_df: rmsle(group_df['sales'], group_df['pred'])).reset_index()
family_result.sort_values(0)




school_data_validation =  validation_data_mod[validation_data_mod.family=="SCHOOL AND OFFICE SUPPLIES"]

plt.scatter( school_data_validation.sales, school_data_validation.pred  )
print(school_data_validation)


test_raw = test_raw.merge(avg_predictions7, on=["store_nbr","family"], how = "left")
test_raw = test_raw.merge(avg_predictions14, on=["store_nbr","family"], how = "left")
test_raw = test_raw.merge(avg_predictions28, on=["store_nbr","family"], how = "left")
test_raw = test_raw.merge(avg_predictions42, on=["store_nbr","family"], how = "left")


test_raw["pred"] = test_raw["MA14"] * test_raw["Dayofweek_factor"] 
test_raw.loc[ test_raw.family=="SCHOOL AND OFFICE SUPPLIES", "pred"] = test_raw.loc[ test_raw.family=="SCHOOL AND OFFICE SUPPLIES",  "MA7"] * test_raw.loc[ test_raw.family=="SCHOOL AND OFFICE SUPPLIES",  "Dayofweek_factor"]

submission_file = test_raw[["id","pred"]]
submission_file = submission_file.rename({"pred":"sales"}, axis= 1)
submission_file
submission_file.to_csv("/kaggle/working/submission.csv", index= False)
plt.show()
