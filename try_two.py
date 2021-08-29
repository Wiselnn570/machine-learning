import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
fname = r'C:\Users\ASUS\Desktop\try.txt'
with open(fname, 'r+', encoding='utf-8') as f:
    s = [i[:-1].split(',') for i in f.readlines()]
    X_raw = [float(s[i][0]) for i in range(len(s))]
    X = [(X_raw[i]-min(X_raw))/(max(X_raw)-min(X_raw)) for i in range(len(X_raw))]
    y_raw = [float(s[i][1]) for i in range(len(s))]
    y = [(y_raw[i]-min(y_raw))/(max(y_raw)-min(y_raw)) for i in range(len(y_raw))]
    X = tf.constant(X,dtype=tf.float32)
    y = tf.constant(y,dtype=tf.float32)
    k = tf.Variable(initial_value=0.)
    b = tf.Variable(initial_value=0.)
    variables = [k,b]
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    for _ in range(10000):
        with tf.GradientTape() as tape:
            y_pred = k*X + b
            loss = tf.reduce_sum(tf.square(y-y_pred))
        grads = tape.gradient(loss,variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads,variables))
    plt.scatter(X,y,c='red')
    plt.plot(X,y_pred,c='blue')
    plt.show()