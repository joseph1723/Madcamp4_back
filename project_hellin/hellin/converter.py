import tensorflow as tf

def convert_model_to_tflite(model_path, tflite_model_path):
    model = tf.keras.models.load_model(model_path)
    1/0
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

# Example usage:
convert_model_to_tflite('classify_workout_model.keras', 'classify_workout_model.tflite')
