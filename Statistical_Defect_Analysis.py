#-----------------------------------------------#-----------------------------------------------
#-----------------------------------------------#-----------------------------------------------
#--------------------------------- PRODCO Defect Project ---------------------------------------
#-----------------------------------------------#-----------------------------------------------
#-----------------------------------------------#-----------------------------------------------


#-----------------------------------------------#-----------------------------------------------
#-------------------------------- 1: Introduction to Data Set ----------------------------------
#-----------------------------------------------#-----------------------------------------------

#----------------------------------------------------------------
#------------------ Data Preparation ---------------------------#
#----------------------------------------------------------------
# importing libraries
import pandas as pd
import numpy as np  # for scientific calculations





# Create function for google drive connection
def download(file):
    from google.colab import auth
    auth.authenticate_user()
    from googleapiclient.discovery import build
    drive_service = build('drive', 'v3')

    file_id = file

    import io
    from googleapiclient.http import MediaIoBaseDownload

    request = drive_service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded, request)
    done = False
    while done is False:
      # _ is a placeholder for a progress object that we ignore.
      # (Our file is small, so we skip reporting progress.)
      _, done = downloader.next_chunk()

    downloaded.seek(0)
    return downloaded




#------------------WIDS_Dataset_2020_Adj.csv-------------------#
# import data from csv:
dwn1 = download('1rhvct0iAiyB5VaI418_m2XSuk_8h3fii')
data = pd.read_csv( dwn1 , sep = "," ,error_bad_lines=False)
data.head()


# Basic information about datatypes
# column names
print("Data:",data.columns)

# variable formats in columns
print("Data:",data.info())





# --- Data cleaning ---
# convert Date column to datetime format
#data["Date"]= pd.to_datetime(data["Date"])

# row cleaning
# Dealing with missing values
# how many null objects are there?
data.isnull().sum()
# delete duplicates in the row
data= data.drop_duplicates(subset=None, keep='first', inplace=False)
# Get rid of the rows with missing values -- we will have 73040 rows
data = data.dropna()

# column cleaning
# list unique values in the column
data['Block_Orientation'].unique()
# I dont Drop Block_Orientation column because I will use it in count calculations (axis=1 represents columns / axis=0 represents rows)
#data.drop(['Block_Orientation'], axis=1, inplace=True)





#----------------------------QC.csv------------------------------#
# import data from csv:
dwn2 = download('1NmSY4Bg673Om2UN5nVLpa_m6gzcqH_I6')
QC = pd.read_csv(dwn2 , sep = ","  )
QC['Defect_Type']=QC['Defect_Type'].str.replace(' ','_')
QC

# variable formats in columns
QC.info()




#------------------------Product_Value.csv-----------------------#
# import data from csv:
dwn3 = download('1KR-VkTTs_Ke2Vmd47gt-TAujsY1ChFZ2')
Product_Value = pd.read_csv(dwn3 , sep = ","  )
Product_Value

# variable formats in columns
Product_Value.info()





#----------------------Maintenance_Costs.csv---------------------#
# import data from csv:
dwn4 = download('17zs23_Yte0bZoBZ9PiYBzYPBsKD2b1MR')
Maintenance_Costs = pd.read_csv(dwn4 , sep = ","  )

# create total cost to fix per hour column by using cost per hour, duration hours and spare parts cost
Maintenance_Costs['Total_cost_to_fix_per_hour'] = (Maintenance_Costs['Cost_per_hour']*Maintenance_Costs['Duration_hours'])+Maintenance_Costs['Spare_parts_cost']
# Maintanence_Cost table
Maintenance_Costs

# note : I exported that file and transposed it to Maintenance_Costs_v3 file by hand




# variable formats in columns
Maintenance_Costs.info()



#--- summary of all data tables ---
# Finding number of rows and columns
print("Number of Rows and Columns data:", data.shape)
print("Number of Rows and Columns M_Costs:", Maintenance_Costs.shape)
print("Number of Rows and Columns Prod_Val:", Product_Value.shape)
print("Number of Rows and Columns QC:", QC.shape)



#----------------------------------------------------------------
#------------------ Data Transformation -------------------------#
#----------------------------------------------------------------

#------------------ Main data ( MERGED ) ------------------------#
# rearranged maintenance_cost data
dwn5 = download('1_1d0kvILL9cvlNjitdWAGT_0temI24Eb')
Maintenance_Costs_v2 = pd.read_csv( dwn5 , sep = "," ,error_bad_lines=False) # Maintenance_Costs_v2 is the transposed version of Maintenance_Costs
Maintenance_Costs_v2.head()



# merging all data tables
merged=pd.merge(data,Product_Value,how='left')  # merging data and Product_Value tables
merged=pd.merge(merged,QC,left_on='Result_Type',right_on='Defect_Type',how='left')  # merging merged and QC tables
merged=merged.drop('Defect_Type',axis=1)  # dropping Defect_Type column in merged data
merged=pd.merge(merged,Maintenance_Costs_v2,left_on='SKU',right_on='SKU',how='left')  # merging new merged table and Maintenance_Costs_v2 table on SKU
merged['Average_Inspection_time'].fillna(0, inplace=True)
merged.reset_index(drop=True)  # deleting index column
merged



# create Result_Type2 that only includes DEFECT-PASS values
merged['Result_Type2']=merged['Result_Type']
merged.Result_Type2.replace({"Defect_1":"DEFECT","Defect_2":"DEFECT","Defect_3":"DEFECT","Defect_4":"DEFECT","Defect_5":"DEFECT"},inplace=True)
merged['Result_Type2'].unique()



# export the data from colab to your pc by converting it to csv
from google.colab import files
merged.to_csv('merged.csv')
files.download('merged.csv')



#-----------------------------------------------#-----------------------------------------------
#------------------------------------ 2: Data Analysis -----------------------------------------
#-----------------------------------------------#-----------------------------------------------

#----------------------Descriptive Summaries---------------------#
# Basic Statistics of object data type columns
merged[['Date','SKU','Zone1_Area','Zone3_Area','Result_Type','Result_Type2']].describe()

# Basic Statistics of numeric data type columns
merged.describe().transpose()



#----------------------Graphical Summaries---------------------#

# import libraries
import matplotlib
import matplotlib.pyplot as plt  # for pilots
import numpy as np  # for scientific calculations
import seaborn as sns # for graphics



# --- Visualization ---
# create a small filtered data for visualization
data_filtered=merged[['Zone1_Dur','Zone2_Dur','Zone3_Dur','Zone1_Temp_Range','Zone2_Temp_Range','Zone3_Temp_Range',
                    'Zone1_Humidity_Range','Zone2_Humidity_Range','Zone3_Humidity_Range','Result_Type']]



# Histogram ---- Stacked Histogram for all data_filtered table
%matplotlib inline

data_filtered.hist(bins=10, figsize=(15,10))
plt.show()





# Scatter matrix
from pandas.plotting import scatter_matrix
attributes = ['Zone1_Dur','Zone2_Dur','Zone3_Dur','Zone1_Temp_Range','Zone2_Temp_Range','Zone3_Temp_Range',
                    'Zone1_Humidity_Range','Zone2_Humidity_Range','Zone3_Humidity_Range']
scatter_matrix(data_filtered[attributes], figsize=(12,8))
plt.show()



# Boxplots using matplotlib
data_filtered.boxplot(figsize=(12,8))
plt.ylabel('Data points')
plt.show()



# Boxplots using seaborm
sns.boxplot(data=data_filtered, orient="h")
plt.xlabel('Data points')
plt.show()




# Density Plot
sns.distplot(merged['Zone1_Dur'], kde=True, bins=30,
             norm_hist=True,
             hist_kws=dict(edgecolor="r", linewidth=2))
plt.xticks(range(0, 30, 40))
plt.ylabel('Density')
plt.show()




# Pie chart for defected product
colors = ['#ffbf80','#ffa64d','#ff8c1a','#ffd9b3','#fff2e6','#d1ccc7']
explode = (0, 0, 0.075, 0, 0, 0.1)
merged['Result_Type'].isnull().sum()
chart_calc = merged.groupby('Result_Type', axis=0).count()
chart_calc
chart_calc['X'].plot(kind='pie', autopct='%1.0f%%' , colors=colors ,shadow=True ,explode=explode)
plt.title('Defected Product Ratio')
plt.ylabel('')
plt.show()



# line plot

fig, ax = plt.subplots(figsize=(15,3))
merged.sort_values(['Date'],ascending=False).groupby(['Date','Result_Type2'])['Block_Orientation'].count().unstack().plot(ax=ax)
plt.legend()



# Heatmap of Correlation
# Correlation matrix
corr_matrix = merged.corr().round(3)
mask = np.zeros_like(corr_matrix,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Setup the matlplotlib figure
f, ax = plt.subplots(figsize=(13,11))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220,10,as_cmap=True)
# Draw the heatmap with mask and correct aspect ratio
sns.heatmap(corr_matrix,mask=mask,cmap=cmap,vmax=.3,center=0,square=True,linewidths=.5,cbar_kws={"shrink":.5})





# Plotting with the FacetGrid() method
g = sns.FacetGrid(data_filtered, col="Result_Type", hue='Zone2_Humidity_Range')
g.map(plt.hist, "Zone2_Temp_Range")
g.add_legend()



# Drop Object Data Type and prepare data for feature selection
merged1 = merged
merged1 = merged.drop(['Date','SKU','Zone1_Area','Zone3_Area','Result_Type','Result_Type2'],axis=1)
merged1.info()


# Other box Plots
merged1.boxplot(figsize=(12,8))
plt.ylabel('Data points')
plt.show()



# Boxplots using seaborm
import seaborn as sns
sns.boxplot(data=merged1)
plt.ylabel('Data points')



# violin plot for Temprature

sns.violinplot(x="Result_Type", y="Z1_Duration_hours", data=merged ,order=["PASS", "Defect_1", "Defect_2", "Defect_3", "Defect_4", "Defect_5"])


sns.violinplot(x="Result_Type", y="Z2_Duration_hours", data=merged ,order=["PASS", "Defect_1", "Defect_2", "Defect_3", "Defect_4", "Defect_5"])


sns.violinplot(x="Result_Type", y="Z3_Duration_hours", data=merged ,order=["PASS", "Defect_1", "Defect_2", "Defect_3", "Defect_4", "Defect_5"])


sns.violinplot(x="Result_Type2", y="Z1_Duration_hours", data=merged ,order=["PASS", "DEFECT"])


sns.violinplot(x="Result_Type2", y="Z2_Duration_hours", data=merged ,order=["PASS", "DEFECT"])


sns.violinplot(x="Result_Type2", y="Z3_Duration_hours", data=merged ,order=["PASS", "DEFECT"])








#----------------------Model Building---------------------#

# Correlation
# Between Zone1 and Zone2 is highly correlated. So if we need we can remove Zone2Position.Other columns seem no correlated
corr_matrix = merged.corr().round(3)
corr_matrix


corr_matrix["Zone1_Humidity_Avg"].sort_values(ascending = False)
corr_matrix["Zone2_Humidity_Avg"].sort_values(ascending = False)
corr_matrix["Zone3_Humidity_Avg"].sort_values(ascending = False)


corr_matrix["Zone1_Temp_Avg"].sort_values(ascending = False)
corr_matrix["Zone2_Temp_Avg"].sort_values(ascending = False)
corr_matrix["Zone3_Temp_Avg"].sort_values(ascending = False)


# --------------------------------------------------------------
# --------------- F E A T U R E  S E L E C T I O N -------------
# --------------------------------------------------------------
# One Hot encoder
#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder(handle_unknown='ignore')
#X_train_enc = enc.fit_transform(X_train)
#X_test_enc = enc.transform(X_test)

# Excluding features with low variance
%matplotlib inline
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
Y = merged.Result_Type # feature extraction
X = merged1


#Variance Threshold
from sklearn.feature_selection import VarianceThreshold
# Set threshold to 0.2
select_features = VarianceThreshold(threshold = 0.2)
select_features.fit_transform(X)
# Subset features
X_subset = select_features.transform(X)
print('Number of features:', X.shape[1])
print('Reduced number of features:',X_subset.shape[1])



# Univariate feature selection
# Chi2 Selector

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

chi2_model = SelectKBest(score_func=chi2, k=4)
X_best_feat = chi2_model.fit_transform(X, Y)



# selected features
print('Number of features:', X.shape[1])
print('Reduced number of features:',X_best_feat.shape[1])



# Feature Selection using Random Forest
from sklearn.ensemble import RandomForestClassifier

# Fit a RandomForest model to the data
model = RandomForestClassifier()
model.fit(X, Y)

# Display the relative importance of each attribute
print(model.feature_importances_)
print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_),X)))



# print bottom 10 features
print(np.sort(model.feature_importances_)[:10])

# print top 10 features
print(np.sort(model.feature_importances_)[-10:])



# split the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                    test_size=0.2, random_state=42)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# combine grid search and cross-validation to find the optimal multiclass logistic regression model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
parameters = {'penalty': ['l2', None],
              'alpha': [1e-07, 1e-06, 1e-05, 1e-04],
              'eta0': [0.01, 0.1, 1, 10]}

# Initialize an SGD logistic regression model
sgd_lr = SGDClassifier(loss='log', learning_rate='constant',
                       eta0=0.01, fit_intercept=True,max_iter=20)
grid_search = GridSearchCV(sgd_lr, parameters,
                               n_jobs=-1, cv=3)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)

# To predict using the optimal model, we apply the following
sgd_lr_best = grid_search.best_estimator_
accuracy = sgd_lr_best.score(X_test, Y_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))



# XGBoost

# Installing XGBoost
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge xgboost

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()






# Kernal SVM

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# Implementing logistic regression using TensorFlow

# Import TensorFlow and specify parameters for the model
import tensorflow as tf
n_features = int(X_train.toarray().shape[1])
learning_rate = 0.001
n_iter = 20

# define placeholders and construct the model by computing the logits
x = tf.placeholder(tf.float32, shape=[None, n_features])
y = tf.placeholder(tf.float32, shape=[None])
W = tf.Variable(tf.zeros([n_features, 1]))
b = tf.Variable(tf.zeros([1]))
logits = tf.add(tf.matmul(x, W), b)[:, 0]
pred = tf.nn.sigmoid(logits)

# get the loss function as well as the measurement of performance the AUC
cost = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
auc = tf.metrics.auc(tf.cast(y, tf.int64), pred)[1]

# define a gradient descent optimizer that searches for the best coefficients by minimizing the loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# initialize the variables and start a TensorFlow session
init_vars = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
sess = tf.Session()
sess.run(init_vars)

# model is trained in a batch manner
batch_size = 1000
import numpy as np
indices = list(range(n_train))
def gen_batch(indices):
  np.random.shuffle(indices)
  for batch_i in range(int(n_train / batch_size)):
    batch_index = indices[batch_i*batch_size:
                             (batch_i+1)*batch_size]
    yield X_train_enc[batch_index], Y_train[batch_index]

# start the training process and print out the loss after each iteration
for i in range(1, n_iter+1):
  avg_cost = 0.
  for X_batch, Y_batch in gen_batch(indices):
    _, c = sess.run([optimizer, cost], feed_dict={x: X_batch.toarray(), y: Y_batch})
    avg_cost += c / int(n_train / batch_size)
print('Iteration %i, training loss: %f' % (i, avg_cost))

# performance check-up on the testing set
auc_test = sess.run(auc,
               feed_dict={x: X_test_enc.toarray(), y: Y_test})
print("AUC of ROC on testing set:", auc_test)




# Big O notation -- This graph will be useful for our presentation
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the style of the plot
plt.style.use('seaborn-whitegrid')

# Creating an array of input sizes
n = 10
x = np.arange(1, n)

# Creating a pandas data frame for popular complexity classes
df = pd.DataFrame({'x': x,
                   'O(1)': 0,
                   'O(n)': x,
                   'O(log_n)': np.log(x),
                   'O(n_log_n)': n * np.log(x),
                   'O(n2)': np.power(x, 2), # Quadratic
                   'O(n3)': np.power(x, 3)}) # Cubic

# Creating labels
labels = ['$O(1) - Constant$',
          '$O(\log{}n) - Logarithmic$',
          '$O(n) - Linear$',
          '$O(n^2) - Quadratic$',
          '$O(n^3) - Cubic$',
          '$O(n\log{}n) - N log n$']

# Plotting every column in dataframe except 'x'
for i, col in enumerate(df.columns.drop('x')):
    print(labels[i], col)
    plt.plot(df[col], label=labels[i])

# Adding a legend
plt.legend()

# Limiting the y-axis
plt.ylim(0,50)

plt.show()



