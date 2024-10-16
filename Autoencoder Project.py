import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense

# 生成隨機數據
data = np.random.rand(100, 5)

# 構建自編碼器模型
input_dim = data.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 訓練自編碼器
autoencoder.fit(data, data, epochs=50, batch_size=10, shuffle=True)

# 提取編碼器部分
encoder = Model(input_layer, encoded)
encoded_data = encoder.predict(data)

# 可視化結果
plt.scatter(encoded_data[:, 0], encoded_data[:, 1])
plt.title('Encoded Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()