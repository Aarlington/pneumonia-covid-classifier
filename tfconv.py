import tensorflow as tf

tf.keras.backend.set_learning_phase(0)
model = tf.keras.models.load_model('./covid19.h5')
export_path = '/models/'

with tf.keras.backend.get_session() as get_session
	tf.saved_model.simple_save(
		sess,
		export_path,
		inputs)
'''
model=tf.keras.models.load_model("./covid19.h5")
converter = pb.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
pb = converter.convert()
open("converted_model_.pb", "wb").write(pb)'''