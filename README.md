# FaceNet_TransferLearning
Transfer weights with keras and facenet.

# Requirements
```bash
pip3 install tensorflow
```

# Usage:

```python
from facenet import FaceNet

# select any number of layers and define the number of classes.
# input shape should be (160, 160, 3). OW there might be negative dimension errors with small images.
fcnet = FaceNet(classes=10, included_layers=1)
print(fcnet.model.summary())
fcnet.model.compile(...)
fcnet.model.fit(...)
```
