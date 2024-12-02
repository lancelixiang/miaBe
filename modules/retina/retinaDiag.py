import tensorflow
from PIL import Image, ImageOps
import numpy as np
import os
import base64
import matplotlib.pyplot as plt


plt.switch_backend('agg')

def main(dir='', img='0 (59).jpeg'):
    # Disable scientific notation for clarity
    plt.cla()
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model(os.path.dirname(__file__) + '/../../models/retina/keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(os.path.dirname(__file__) + f'/../../upload/{dir}/' + img)

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    # image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    # print(prediction)

    # determining predicted result
    pred_new = prediction[0]
    pred = max(pred_new)

    # print(pred_new)
    index = pred_new.tolist().index(pred)

    #plot the graph


    # x-coordinates of left sides of bars
    left = [1, 2, 3, 4, 5]

    # heights of bars
    height = pred_new.tolist()
    new_height = []
    for i in height:
        new_height.append(round(i, 2) * 100)

    # print(height)
    # print("+++++++++++++++")
    # print("new height:",new_height)
    tick_label = ['NO_DR', 'mild', 'moderate', 'sever', 'proliferative']

    # plotting a bar chart
    plt.bar(left, new_height, tick_label=tick_label,
            width=0.8, color=['red', 'green'])

    # naming the x-axis
    plt.xlabel('x - axis')
    # naming the y-axis
    plt.ylabel('y - axis')
    # plot title
    plt.title('Diabetic Retinopathy')

    # function to show the plot
    plt.savefig(os.path.dirname(__file__) + '/output/graph.png')
    result = []
    with open(os.path.dirname(__file__) + '/output/graph.png', 'rb') as f:
        img_data = f.read()
    result.append(base64.b64encode(img_data).decode('utf-8'))
    result.append("-")
    # plt.savefig(path_ + '/output1/graph2.png')
    # plt.show()
    # plt.close()
    if index == 0:
        result.append("无糖尿病")
    elif index == 1:
        result.append("轻度糖尿病")
    elif index == 2:
        result.append("中度糖尿病")
    elif index == 3:
        result.append("重度糖尿病")
    elif index == 4:
        result.append("激增性重度糖尿病")

    accuracy = round(pred, 2)
    result.append("-")
    result.append(accuracy * 100)

    return result
