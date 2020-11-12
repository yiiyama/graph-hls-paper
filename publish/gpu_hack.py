try:
    import tensorflow.keras as keras
except ImportError:
    import keras

if hasattr(keras.backend, 'tensorflow_backend'):
    import tensorflow as tf
    
    if float(tf.__version__[:tf.__version__.rfind('.')]) > 2.:
        # Redefine _get_available_gpus function of the backend
        tfback = keras.backend.tensorflow_backend
    
        def _get_available_gpus():
            """Get a list of available gpu devices (formatted as strings).
        
            # Returns
                A list of available GPU devices.
            """
            #global _LOCAL_DEVICES
            if tfback._LOCAL_DEVICES is None:
                devices = tf.config.list_logical_devices()
                tfback._LOCAL_DEVICES = [x.name for x in devices]
            return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
        
        tfback._get_available_gpus = _get_available_gpus
