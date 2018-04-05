import tensorflow as tf
import keras as K
import keras.backend as kb
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE


df = pd.read_csv("datasets/criminal_train.csv")

#split in test and train
train_X,test_X,train_Y,test_Y = train_test_split(df.drop(['Criminal'],axis=1),
                                                 df['Criminal'], test_size = 0.15)


#before oversampling
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

df['Criminal'].value_counts().plot(ax=axes[0],kind='bar', title='Criminal').set_xlabel(df['Criminal'].value_counts())
train_Y[:].value_counts().plot(ax=axes[1],kind='bar',title ='train').set_xlabel(train_Y[:].value_counts())
test_Y[:].value_counts().plot(ax=axes[2],kind='bar',title ='test').set_xlabel(test_Y[:].value_counts())



#create validation set
#train_Y = pd.DataFrame(train_Y.values,columns=['Criminal'])

train_X,val_X,train_Y,val_Y = train_test_split(train_X,train_Y, test_size = 0.15)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
train_Y[:].value_counts().plot(ax=axes[1],kind='bar',title ='train').set_xlabel(train_Y[:].value_counts())
val_Y[:].value_counts().plot(ax=axes[0],kind='bar',title ='val').set_xlabel(val_Y[:].value_counts())



print(train_X.shape,train_Y.shape,val_X.shape,val_Y.shape)
train_Y = train_Y.reshape(-1,1)
print(train_Y.shape)


#oversampling
sm = SMOTE(ratio = 1)
train_X,train_Y = sm.fit_sample(train_X,train_Y)





bins,counts = np.unique(train_Y,return_counts=True)
print(train_X.shape,train_Y.shape)
print(np.asarray((bins, counts)).T)



train_X = train_X[:,1:]
train_Y = train_Y.reshape(-1,1)
print(train_X.shape)


def finalmodel(input_shape):
    
    X_input = K.layers.Input(input_shape)
    X = X_input
    
    X = K.layers.Dropout(0.2)(X)
    X = K.layers.Dense(units = 25, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer ='zeros')(X)
                          #,kernel_regularizer = K.regularizers.l2(0.01))(X)
    X = K.layers.Dropout(0.2)(X)
    
    X = K.layers.Dense(units = 10, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer ='zeros')(X)
                         # ,kernel_regularizer = K.regularizers.l2(0.01))(X)
    X = K.layers.Dropout(0.2)(X)
    
    X = K.layers.Dense(units = 1, activation = 'sigmoid',kernel_initializer = 'random_uniform', bias_initializer = 'zeros')(X)
                         # ,kernel_regularizer = K.regularizers.l2(0.01))(X)
    
    model = K.models.Model(inputs = X_input, outputs = X, name = "finalmodel");
    
    return model


def matthews_correlation(y_true, y_pred):
    y_pred_pos = kb.round(kb.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = kb.round(kb.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = kb.sum(y_pos * y_pred_pos)
    tn = kb.sum(y_neg * y_pred_neg)

    fp = kb.sum(y_neg * y_pred_pos)
    fn = kb.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = kb.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + kb.epsilon())


final_model = finalmodel((70,))

adam = K.optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#sgd  = K.optimizers.sgd(lr=0.0001,decay=1e-6)
final_model.compile(optimizer='adam',loss= K.losses.mean_squared_logarithmic_error,metrics=[matthews_correlation])



print(np.unique(train_Y))

tbCallback = K.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0, batch_size=1024, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

final_model.fit(train_X,train_Y,epochs = 500,batch_size=1024,callbacks=[tbCallback])

final_model.summary()


perid = test_X[test_X.columns[0:1]];
temp_val_X = val_X[val_X.columns[1:]]
print(test_X.shape,val_Y.shape)
temp_test_X = test_X[test_X.columns[1:]]
print(temp_test_X.shape,val_Y.shape)


final_model.evaluate(x=temp_val_X,y=val_Y)


final_model.evaluate(x=temp_test_X,y=test_Y)

final_model.predict(x=temp_test_X)

df = pd.read_csv("datasets/criminal_test.csv")


perid = df[df.columns[:1]]
final_test_X = df[df.columns[1:]] 
print(final_test_X.shape)


output  = final_model.predict(x=final_test_X)
output = [1 if x>0.85 else 0 for x in output]
output = pd.DataFrame(output,columns=["Criminal"])
answer = pd.concat([perid,output],axis=1)
answer.to_csv("out.csv",index=False)
