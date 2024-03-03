#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

#  Loading and preprocess the data
folder_path = r"C:\Users\LEN\Dropbox\PC\Desktop\datafiles"
file_names = os.listdir(folder_path)

# lists to store the data
training_data = []
testing_data = []

# Extracting the csv files
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    if 'PID' not in df.columns:
        continue  

    # Group the data by PID
    grouped = df.groupby('PID')

    # Process each PID in the clip
    for pid, group in grouped:
        frames = group['FRAMEID'].values
        poses = group['POSES'].str.replace('[', '').str.replace(']', '')
        pose_list = poses.str.split(',', expand=True)
        neck_x = pose_list[2].astype(float)
        neck_y = pose_list[3].astype(float)
        if len(neck_x) >= 2 and len(neck_y) >= 2:
            orientation = np.degrees(np.arctan2(np.diff(neck_y), np.diff(neck_x)))
            orientation = np.concatenate(([orientation[0]], orientation))
        else:
            orientation = np.zeros(len(neck_x))
        # Calculate velocity
        if len(neck_x) >= 2 and len(neck_y) >= 2:
            velocity = np.sqrt(np.diff(neck_x)**2 + np.diff(neck_y)**2)
            velocity = np.concatenate(([velocity[0]], velocity))
        else:
            velocity = np.zeros(len(neck_x))

        #missing frame IDs 
        all_frame_ids = np.arange(frames.min(), frames.max() + 1)
        missing_frame_ids = np.setdiff1d(all_frame_ids, frames)
        missing_neck_x = np.repeat(np.nan, len(missing_frame_ids))
        #missing_neck_x = np.nan_to_num(missing_neck_x, nan=0.0)
        missing_neck_y = np.repeat(np.nan, len(missing_frame_ids))
        #missing_neck_y = np.nan_to_num(missing_neck_y, nan=0.0)
        missing_orientation = np.repeat(np.nan, len(missing_frame_ids))
        #missing_orientation = np.nan_to_num(missing_orientation, nan=0.0)
        missing_velocity = np.repeat(np.nan, len(missing_frame_ids))
        #missing_velocity = np.nan_to_num(missing_velocity, nan=0.0)

        # Append missing frame IDs  to the trajectory data
        frames = np.concatenate([frames, missing_frame_ids])
        neck_x = np.concatenate([neck_x, missing_neck_x])
        neck_y = np.concatenate([neck_y, missing_neck_y])
        orientation = np.concatenate([orientation, missing_orientation])
        velocity = np.concatenate([velocity, missing_velocity])

        combined_frame_ids = np.concatenate((frames, missing_frame_ids))
        combined_neck_x = np.concatenate((neck_x, missing_neck_x))
        combined_neck_y = np.concatenate((neck_y, missing_neck_y))
        combined_orientation = np.concatenate((orientation, missing_orientation))
        combined_velocity = np.concatenate((velocity, missing_velocity))

        # Sort the combined arrays based on frame IDs
        sorted_indices = np.argsort(combined_frame_ids)
        sorted_frame_ids = combined_frame_ids[sorted_indices]
        sorted_neck_x = combined_neck_x[sorted_indices]
        sorted_neck_y = combined_neck_y[sorted_indices]
        sorted_orientation = combined_orientation[sorted_indices]
        sorted_velocity = combined_velocity[sorted_indices]

        # DataFrame for the PID's trajectory data
        trajectory_df = pd.DataFrame({
            'FRAMEID': sorted_frame_ids,
            'neck_x': sorted_neck_x,
            'neck_y': sorted_neck_y,
            'orientation': sorted_orientation,
            'velocity': sorted_velocity
        })
        trajectory_df = trajectory_df.drop_duplicates(subset=['FRAMEID'], keep='last')

        interpolate_columns = ['neck_x', 'neck_y', 'orientation', 'velocity']
        trajectory_df[interpolate_columns] = trajectory_df[interpolate_columns].interpolate(method='linear')

        print(trajectory_df.to_string(index=False))

        num_windows = len(frames) // 50

        # Split the trajectory into windows of frames
        windows = [trajectory_df.iloc[i * 50:(i + 1) * 50] for i in range(num_windows)]

        if len(windows) < 2:
            continue  

        # Split the windows into training and testing sets based on PIDs
        train_windows, test_windows = train_test_split(windows, test_size=0.1, random_state=42)
        if len(train_windows) > 0:
            training_data.extend(train_windows)
        if len(test_windows) > 0:
            testing_data.extend(test_windows)
#training
train_frames = []
train_targets = []
for window in training_data:
    frames = window.index
    points = window[['neck_x', 'neck_y', 'orientation', 'velocity']].values
    train_frames.append(frames)
    train_targets.append(points)

train_frames = np.array(train_frames)
train_targets = np.array(train_targets)

#testing data
test_frames = []
test_targets = []
for window in testing_data:
    frames = window.index
    points = window[['neck_x', 'neck_y', 'orientation', 'velocity']].values
    test_frames.append(frames)
    test_targets.append(points)

test_frames = np.array(test_frames)
test_targets = np.array(test_targets)

# Normalize the input data
scaler = StandardScaler()
train_targets_scaled = scaler.fit_transform(train_targets.reshape(-1, 4)).reshape(train_targets.shape)
test_targets_scaled = scaler.transform(test_targets.reshape(-1, 4)).reshape(test_targets.shape)

#LSTM model
input_shape = (50, 4)

# Encoder
encoder_input = tf.keras.layers.Input(shape=input_shape)
encoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True)(encoder_input)
encoder_attention = tf.keras.layers.Attention()([encoder_lstm, encoder_lstm])

# Reconstruction Decoder
reconstruction_decoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True)(encoder_attention)
reconstruction_decoder_attention = tf.keras.layers.Attention()([reconstruction_decoder_lstm, reconstruction_decoder_lstm])
reconstruction_output = tf.keras.layers.Dense(4)(reconstruction_decoder_attention)

# Imputation Decoder
imputation_input = tf.keras.layers.Input(shape=input_shape)
imputation_lstm = tf.keras.layers.LSTM(256, return_sequences=True)(imputation_input)

# Self-Attention Mechanism
num_heads = 4  # Number of attention heads
hidden_units = 256  # Hidden units for each head
attention_heads = []
for _ in range(num_heads):
    attention_heads.append(tf.keras.layers.Attention()([imputation_lstm, imputation_lstm]))
concatenated_attention_heads = tf.keras.layers.Concatenate()(attention_heads)
imputation_output_lstm = tf.keras.layers.LSTM(hidden_units, return_sequences=True)(concatenated_attention_heads)
imputation_output = tf.keras.layers.Dense(4)(imputation_output_lstm)

# Model
model = tf.keras.models.Model(inputs=[encoder_input, imputation_input], outputs=[reconstruction_output, imputation_output])
model.compile(loss='mse', optimizer='adam')
model.fit([train_targets_scaled, train_targets_scaled], [train_targets_scaled, train_targets_scaled], epochs=500, batch_size=256)

reconstruction_predictions, imputation_predictions = model.predict([test_targets_scaled, test_targets_scaled])

#Rescale the predicted data
reconstruction_predictions_rescaled = scaler.inverse_transform(reconstruction_predictions.reshape(-1, 4)).reshape(
    reconstruction_predictions.shape)
imputation_predictions_rescaled = scaler.inverse_transform(
    imputation_predictions.reshape(-1, 4)).reshape(imputation_predictions.shape)

for i in range(len(test_frames)):
    print("Predicted Points for Window:", i + 1)
    print("Reconstruction Predictions:")
    print(reconstruction_predictions_rescaled[i])
    print("Imputation Predictions:")
    print(imputation_predictions_rescaled[i])
    print("Ground Truth Points:")
    print(test_targets[i])
    print()
    print()

