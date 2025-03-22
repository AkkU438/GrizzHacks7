import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd

#load dataset
train_df = pd.read_csv()
test_df = pd.read_csv()

#convert dataset into tf dataset
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df)
