#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 18:38:40 2026

@author: marleeyork
"""
import numpy as np
import tensorflow as tf

# Creating random data points
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(mean=[0,3], 
                                                 cov=[[1, 0.5], [0.5, 1]], 
                                                 size=num_samples_per_class)

positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)

# Stack into a single array
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

# Generate the target labels
targets = np.vstack(
    (
        np.zeros((num_samples_per_class, 1), dtype="float32"),
        np.ones((num_samples_per_class, 1), dtype="float32"),
    )
)

# Input dimension is 2D
input_dim = 2
output_dim = 1

# Generate initial weights and intercept
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

# Define forward pass function
def model(inputs, W, b):
    return tf.matmul(inputs, W) + b

# Define the loss function (MSE)
def mean_squared_error(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)

# Now we train the model
learning_rate = 0.1

# Wrap function in tf.function decorator to speed it up
@tf.function(jit.compile=True)
def training_step(inputs, targets, W, b):
    # Perform a forward pass inside of gradient tape scope
    with tf.GradientTape() as tape:
        predictions = model(inputs, W, b)
        loss = mean_squared_error(targets, predictions)
    # Retrieve the gradient of the loss with regard to the weights
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss,[W,b])
    # Updates the weights
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

# Train the model
for step in range(40):
    loss = training_step(inputs, targets, W, b)
    print(f"Loss at step {step}: {loss:.4f}")

predictions = model(inputs, W, b)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()

x = np.linspace(-1, 4, 100)
# This is our line's equation.
y = -W[0] / W[1] * x + (0.5 - b) / W[1]
# Plots our line (`"-r"` means "plot it as a red line")
plt.plot(x, y, "-r")
# Plots our model's predictions on the same plot
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
    
    
    
    