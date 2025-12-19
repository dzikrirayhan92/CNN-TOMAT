import gradio as gr
import pathlib
import random
import tensorflow as tf

from PIL import Image
from timeit import default_timer as timer
from keras.models import load_model


model = load_model('model-tomatos1112-vgg_TA2.h5')
label2id = {'Bukan Tomat': 0, 'Matang': 1, 'Mentah': 2, 'Setengah Matang': 3}
class_names = list(label2id.keys())

def predict(img):
    start = timer()

    img = img.resize((224, 224))
    img_aug = tf.keras.layers.Rescaling(1., input_shape=(224, 224, 3))
    output = tf.expand_dims(img_aug(img), 0)
    pred_prob = tf.nn.softmax(model.predict(output))
    pred_dict = {class_names[i]:float(tf.squeeze(pred_prob)[i]) for i in range(len(class_names))}

    pred_time = round(timer() - start, 5)

    return pred_dict, pred_time



title = 'Prediksi tingkat kematangan tomat'
description = ''

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=[gr.Label(num_top_classes=4, label='Predictions'),
             gr.Number(label="Prediction time (s)")],
    description=description,
    title=title,
    allow_flagging='never',
)

demo.launch(debug=True)