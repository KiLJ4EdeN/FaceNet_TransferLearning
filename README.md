# FaceNet_TransferLearning
Transfer weights with keras and facenet.

# Requirements
```bash
pip3 install tensorflow
```

# Usage:

## Clone the Repo:
```bash
git clone https://github.com/KiLJ4EdeN/FaceNet_TransferLearning
cd FaceNet_TransferLearning
```

## Create the FaceNet model and a dataset.
```python
# train.py
from facenet import FaceNet
import numpy as np

# create data
import numpy as np
from tensorflow.keras.utils import to_categorical

X = np.random.rand(1000, 160, 160, 3)
y = np.random.randint(0, 10, size=(1000))
y = to_categorical(y)
print(X.shape)
print(y.shape)

# select any number of layers and define the number of classes.
# input shape should be (160, 160, 3). Minumum is (75, 75, 3).
facenet = FaceNet(input_shape=(160, 160, 3), classes=10, included_layers=1)
print(facenet.summary())
```

## Fit the model.
```python
facenet.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
facenet.fit(x=X, y=y, epochs=10, batch_size=128)
```
