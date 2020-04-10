import tensorflow as tf
model=tf.keras.models.load_model("./covid19.h5")
converter = pb.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
pb = converter.convert()
open("converted_model_.pb", "wb").write(pb)