# FaceNet_TransferLearning
Transfer weights with keras and facenet.

# Usage:

```python
# select any number of layers and define the number of classes.
# input shape should be (160, 160, 3). OW there might be negative dimension errors with small images.
fcnet = FaceNet(classes=10, included_layers=1)
print(fcnet.model.summary())
model.compile(...)
model.fit(...)
```
