import tensorflow
import data
import matplotlib.pyplot

data_train, data_test, label_train, label_test = data.load_data()

print(data_train.shape)
print(label_train.shape)

data_train = data_train.reshape((data_train.shape[0]*128, 128, 128))
data_test = data_test.reshape((data_test.shape[0]*128, 128, 128))
label_train = data_train.reshape((label_train.shape[0]*128, 128, 128))
label_test = data_test.reshape((label_test.shape[0]*128, 128, 128))

print(data_train.shape)
print(label_train.shape)

# matplotlib.pyplot.matshow(data_train[0,60])
# matplotlib.pyplot.show()

base_model = tensorflow.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]

layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tensorflow.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tensorflow.random_normal_initializer(0., 0.02)

  result = tensorflow.keras.Sequential()
  result.add(
      tensorflow.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tensorflow.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tensorflow.keras.layers.Dropout(0.5))

  result.add(tensorflow.keras.layers.ReLU())

  return result

up_stack = [
    upsample(512, 3),  # 4x4 -> 8x8
    upsample(256, 3),  # 8x8 -> 16x16
    upsample(128, 3),  # 16x16 -> 32x32
    upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  # This is the last layer of the model
  last = tensorflow.keras.layers.Conv2DTranspose(
    output_channels, 3, strides=2,
    padding='same', activation='softmax')  #64x64 -> 128x128

  inputs = tensorflow.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tensorflow.keras.layers.Concatenate()
    x = concat([x, skip])

  x = last(x)

  return tensorflow.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CHANNELS = 2
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# plot model to file model.png
tensorflow.keras.utils.plot_model(model, show_shapes=True)

# def create_mask(pred_mask):
#   pred_mask = tf.argmax(pred_mask, axis=-1)
#   pred_mask = pred_mask[..., tf.newaxis]
#   return pred_mask[0]
#
# def show_predictions(dataset=None, num=1):
#   if dataset:
#     for image, mask in dataset.take(num):
#       pred_mask = model.predict(image)
#       display([image[0], mask[0], create_mask(pred_mask)])
#   else:
#     display([sample_image, sample_mask,
#              create_mask(model.predict(sample_image[tf.newaxis, ...]))])

EPOCHS = 20
VAL_SUBSPLITS = 4
BATCH_SIZE = 64
VALIDATION_STEPS = BATCH_SIZE//VAL_SUBSPLITS
STEPS_PER_EPOCH = data_train.shape[0]

model_history = model.fit(
    data_train,
    label_train,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    validation_data=(data_test, label_test),
)
