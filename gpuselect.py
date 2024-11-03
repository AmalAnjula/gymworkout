import tensorflow as tf

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('CPU')
print(physical_devices)
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and being used")
else:
    print("GPU is not available, using CPU")