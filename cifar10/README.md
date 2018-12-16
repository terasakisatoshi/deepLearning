# cifar10

# cifar10_cnn.py

```
$ python cifar10_cnn.py
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        896       
_________________________________________________________________
dropout (Dropout)            (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 64)        18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 4096)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               2097664   
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5130      
_________________________________________________________________
activation (Activation)      (None, 10)                0         
=================================================================
Total params: 2,159,114
Trainable params: 2,159,114
Non-trainable params: 0
_________________________________________________________________
Train on 40000 samples, validate on 10000 samples
2018-12-16 22:09:50.542152: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-12-16 22:09:50.818824: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.911
pciBusID: 0000:03:00.0
totalMemory: 7.93GiB freeMemory: 7.81GiB
2018-12-16 22:09:51.034872: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 1 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.8225
pciBusID: 0000:04:00.0
totalMemory: 7.92GiB freeMemory: 7.14GiB
2018-12-16 22:09:51.036045: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0, 1
2018-12-16 22:09:51.672023: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-12-16 22:09:51.672068: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 1 
2018-12-16 22:09:51.672078: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N Y 
2018-12-16 22:09:51.672084: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 1:   Y N 
2018-12-16 22:09:51.672479: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7535 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:03:00.0, compute capability: 6.1)
2018-12-16 22:09:51.672870: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 6883 MB memory) -> physical GPU (device: 1, name: GeForce GTX 1080, pci bus id: 0000:04:00.0, compute capability: 6.1)
Epoch 1/12
40000/40000 [==============================] - 6s 161us/step - loss: 1.7179 - acc: 0.3815 - val_loss: 1.4275 - val_acc: 0.5022
Epoch 2/12
40000/40000 [==============================] - 5s 125us/step - loss: 1.2314 - acc: 0.5665 - val_loss: 1.1200 - val_acc: 0.6048
Epoch 3/12
40000/40000 [==============================] - 5s 126us/step - loss: 1.0301 - acc: 0.6390 - val_loss: 0.9849 - val_acc: 0.6507
Epoch 4/12
40000/40000 [==============================] - 5s 125us/step - loss: 0.8898 - acc: 0.6892 - val_loss: 0.9730 - val_acc: 0.6701
Epoch 5/12
40000/40000 [==============================] - 5s 124us/step - loss: 0.7971 - acc: 0.7232 - val_loss: 0.8695 - val_acc: 0.7027
Epoch 6/12
40000/40000 [==============================] - 5s 124us/step - loss: 0.6996 - acc: 0.7566 - val_loss: 0.8871 - val_acc: 0.7072
Epoch 7/12
40000/40000 [==============================] - 5s 125us/step - loss: 0.6243 - acc: 0.7841 - val_loss: 0.8042 - val_acc: 0.7335
Epoch 8/12
40000/40000 [==============================] - 5s 124us/step - loss: 0.5600 - acc: 0.8090 - val_loss: 0.8417 - val_acc: 0.7312
Epoch 9/12
40000/40000 [==============================] - 5s 124us/step - loss: 0.5020 - acc: 0.8270 - val_loss: 0.9939 - val_acc: 0.7318
Epoch 10/12
40000/40000 [==============================] - 5s 123us/step - loss: 0.4535 - acc: 0.8425 - val_loss: 0.8663 - val_acc: 0.7385
Epoch 11/12
40000/40000 [==============================] - 5s 124us/step - loss: 0.4121 - acc: 0.8588 - val_loss: 0.8880 - val_acc: 0.7409
Epoch 12/12
40000/40000 [==============================] - 5s 125us/step - loss: 0.3835 - acc: 0.8713 - val_loss: 1.1019 - val_acc: 0.7291
loss 1.15116297621727
accuracy 0.7263
```


# cifar10_cnn_augment.py

```
$ python cifar10_cnn_augment.py 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        896       
_________________________________________________________________
dropout (Dropout)            (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 64)        18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 4096)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               2097664   
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5130      
_________________________________________________________________
activation (Activation)      (None, 10)                0         
=================================================================
Total params: 2,159,114
Trainable params: 2,159,114
Non-trainable params: 0
_________________________________________________________________
2018-12-16 22:19:26.236817: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-12-16 22:19:26.555800: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.911
pciBusID: 0000:03:00.0
totalMemory: 7.93GiB freeMemory: 7.81GiB
2018-12-16 22:19:26.781849: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 1 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.8225
pciBusID: 0000:04:00.0
totalMemory: 7.92GiB freeMemory: 7.04GiB
2018-12-16 22:19:26.782953: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0, 1
2018-12-16 22:19:27.415574: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-12-16 22:19:27.415625: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 1 
2018-12-16 22:19:27.415636: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N Y 
2018-12-16 22:19:27.415642: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 1:   Y N 
2018-12-16 22:19:27.416069: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7535 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:03:00.0, compute capability: 6.1)
2018-12-16 22:19:27.416464: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 6787 MB memory) -> physical GPU (device: 1, name: GeForce GTX 1080, pci bus id: 0000:04:00.0, compute capability: 6.1)
Epoch 1/15
312/312 [==============================] - 37s 119ms/step - loss: 1.8004 - acc: 0.3541 - val_loss: 1.3632 - val_acc: 0.5226
Epoch 2/15
312/312 [==============================] - 35s 113ms/step - loss: 1.3980 - acc: 0.5037 - val_loss: 1.1286 - val_acc: 0.6004
Epoch 3/15
312/312 [==============================] - 36s 115ms/step - loss: 1.2225 - acc: 0.5669 - val_loss: 1.0416 - val_acc: 0.6364
Epoch 4/15
312/312 [==============================] - 36s 116ms/step - loss: 1.1201 - acc: 0.6078 - val_loss: 0.9265 - val_acc: 0.6749
Epoch 5/15
312/312 [==============================] - 36s 117ms/step - loss: 1.0445 - acc: 0.6364 - val_loss: 0.9031 - val_acc: 0.6855
Epoch 6/15
312/312 [==============================] - 36s 117ms/step - loss: 0.9861 - acc: 0.6590 - val_loss: 0.8532 - val_acc: 0.6980
Epoch 7/15
312/312 [==============================] - 35s 112ms/step - loss: 0.9429 - acc: 0.6727 - val_loss: 0.8214 - val_acc: 0.7136
Epoch 8/15
312/312 [==============================] - 36s 114ms/step - loss: 0.9084 - acc: 0.6833 - val_loss: 0.8892 - val_acc: 0.6988
Epoch 9/15
312/312 [==============================] - 36s 115ms/step - loss: 0.8823 - acc: 0.6960 - val_loss: 0.7405 - val_acc: 0.7533
Epoch 10/15
312/312 [==============================] - 36s 115ms/step - loss: 0.8583 - acc: 0.7046 - val_loss: 0.7950 - val_acc: 0.7288
Epoch 11/15
312/312 [==============================] - 35s 112ms/step - loss: 0.8423 - acc: 0.7088 - val_loss: 0.7602 - val_acc: 0.7415
Epoch 12/15
312/312 [==============================] - 36s 115ms/step - loss: 0.8266 - acc: 0.7177 - val_loss: 0.6942 - val_acc: 0.7602
Epoch 13/15
312/312 [==============================] - 35s 113ms/step - loss: 0.8144 - acc: 0.7189 - val_loss: 0.7185 - val_acc: 0.7571
Epoch 14/15
312/312 [==============================] - 34s 110ms/step - loss: 0.8098 - acc: 0.7220 - val_loss: 0.7113 - val_acc: 0.7685
Epoch 15/15
312/312 [==============================] - 35s 111ms/step - loss: 0.7974 - acc: 0.7271 - val_loss: 0.7397 - val_acc: 0.7519
loss 0.7560383714199066
accuracy 0.7536
```