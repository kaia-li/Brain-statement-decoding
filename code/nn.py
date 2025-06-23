import pandas as pd
import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nilearn as nl
import sklearn
import datetime as dt
import os
import glob

import nibabel as nib
import nilearn.plotting as plotting
import hcp_utils as hcp
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiMapsMasker
from nilearn import plotting

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib as lib
lib.use('TKAgg')

# Load the CSV data into a DataFrame 
# mmp/n/combined17m.csv 
# 17/combined_nonorm_17.csv 
# mmp/n/combined_nonorm_mmp.csv 
# 7_n/combined_nonorm_7.csv
data = pd.read_csv("/Users/kaiali/Documents/HCP/HCP_all_data/New/mmp/comb_n.csv",header=0,index_col=0)
print(data.dtypes)
data.iloc[:,-1].replace(('Language','Motion','Loss','Reward','Relation','Emotion','Memory','Social'),(0,1,2,3,4,5,6,7),inplace=True)
# print(data.head(10))
#data.iloc[:, -1] = data.iloc[:, -1].astype('category')


# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42, stratify=data.iloc[:, -1])




# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)

 # Apply PCA to reduce the dimensionality of the features
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_norm) #X_train_norm
X_val_pca = pca.transform(X_val_norm) #X_test_norm 

# Define the neural network model

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=X_train_norm.shape[1],kernel_regularizer=tf.keras.regularizers.l2(0.01)),#norm
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(8, activation='softmax')
])


# Compile the model with a categorical loss function and optimizer

initial_lr = 0.001

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=initial_lr),metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss',patience=1)



# Train the model on the training data norm
history = model.fit(X_train_norm, pd.get_dummies(y_train), epochs=300, batch_size=32, validation_split=0.1,callbacks=[early_stop])

# Evaluate the accuracy of the model on the testing data
accuracy = model.evaluate(X_val_norm, pd.get_dummies(y_val))[1]
print('Val accuracy:', accuracy)

path_to_dir = '/Users/kaiali/Documents/HCP/NEOM/mat/v3/nif'
nii_files = glob.glob(os.path.join(path_to_dir, "*.nii"))
output_fig_dir = '/Users/kaiali/Documents/HCP/NEOM/mat/v3/fig'
output_csv_dir = '/Users/kaiali/Documents/HCP/NEOM/mat/v3/matrix'
# new_data = pd.read_csv("/Users/kaiali/Documents/HCP/NEOM/mat/mmp/106.csv",header=0,index_col=0)
# print(new_data.shape)
probs_list = []

for file in nii_files:
    X = nib.load(file).get_data()
    Xp = hcp.parcellate(X, hcp.mmp)  
    labels = hcp.mmp.labels
    regions = [labels[key] for key in labels if key != 0]
    #print(regions)
    new_data = pd.DataFrame(Xp)
    columns=regions
    
    loaded_model = tf.keras.models.load_model("/Users/kaiali/Documents/HCP/NEOM/mat/new/new_model.h5")
    predicted_classes = loaded_model.predict(new_data)
    probs = loaded_model.predict_proba(new_data)
    probs_list.append(probs.tolist())

    prob_df = pd.DataFrame(probs[19:715], columns=['Language','Motion','Loss','Reward','Relation','Emotion','Memory','Social'])
    f = os.path.splitext(os.path.basename(file))[0][:6]
    output_file = os.path.join(output_csv_dir, f"{f}.csv")
    prob_df.to_csv(output_file, index=False)
    probs_df = prob_df.transpose()
    plt.figure(figsize=(10, 8))
    sns.heatmap(probs_df, cmap='coolwarm', vmin=0, vmax=1)
    
    
    plt.title(os.path.join("prob_"+f"{f}.png"))
    plt.xlabel('Time')
    plt.ylabel('Class')


    plot_filename = os.path.join(output_fig_dir, f"{f}.png")
    plt.savefig(plot_filename)
    #plt.show()

probs_dfs = []

# Create a DataFrame for each input file and add it to the list of DataFrames
for file_probs in probs_list:
    probs_df = pd.DataFrame(file_probs, columns=['Language','Motion','Loss','Reward','Relation','Emotion','Memory','Social']).transpose()
    probs_dfs.append(probs_df)

# Concatenate all the DataFrames along the x-axis (samples) and compute the average of each cell
average_probs_df = pd.concat(probs_dfs, axis=1)
average_probs_dfs = average_probs_df.groupby(level=0, axis=1).mean()
average_probs_dfs.transpose().to_csv('/Users/kaiali/Documents/HCP/NEOM/mat/v3/matrix/new_average_trans_.csv',index=False)
# Plot the heatmap of the average probabilities
plt.figure(figsize=(10, 8))
sns.heatmap(average_probs_dfs, cmap='coolwarm', vmin=0, vmax=1)

# Set the plot title and axis labels
plt.title("New Average Predicted Probabilities")
plt.xlabel('Time')
plt.ylabel('Class')

# Save the plot as an image file in the specified output directory
avg_plot = os.path.join(output_fig_dir, "new_average_probs.png")
plt.savefig(avg_plot)

# Show the plot
#plt.show()









"""

y_pred = model.predict(X_val_norm)
y_pred_categorical = np.argmax(y_pred, axis=1)
# cm = confusion_matrix(y_train, y_pred)
cm = confusion_matrix(y_val, y_pred_categorical)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('New Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_train)))
indices = range(len(cm))
classes = ['Language_Story_Maths','Motor_Avg','Punish_Reward','Reward_Punish','Relational_Relational_Match','Emo_face-WM_face_AVG','WM_2BK_0BK','Social_Tom_Random']
plt.xticks(indices,['Language_Story_Maths','Motor_Avg','Punish_Reward','Reward_Punish','Relational_Relational_Match','Emo_face-WM_face_AVG','WM_2BK_0BK','Social_Tom_Random'],rotation=45)
plt.yticks(indices,['Language_Story_Maths','Motor_Avg','Punish_Reward','Reward_Punish','Relational_Relational_Match','Emo_face-WM_face_AVG','WM_2BK_0BK','Social_Tom_Random'],rotation=45)

plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.show() 

class_report = classification_report(y_val, y_pred_categorical,target_names=classes)
print(class_report)

# Save the model
model.save('new_model.h5') 
"""
