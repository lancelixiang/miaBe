import tensorflow
from PIL import Image, ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import random
import string
from pathlib import Path

plt.switch_backend('agg')

def generate_random_string(length=10):
    # 定义可以使用的字符集（字母和数字）
    characters = string.ascii_letters + string.digits
    # 使用random.choices从字符集中随机选择字符
    random_string = ''.join(random.choices(characters, k=length))
    return random_string
 

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

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    # determining predicted result
    pred_new = prediction[0]
    pred = max(pred_new)

    # print(pred_new)
    index = pred_new.tolist().index(pred)

    # x-coordinates of left sides of bars
    left = [1, 2, 3, 4, 5]

    # heights of bars
    height = pred_new.tolist()
    new_height = []
    for i in height:
        new_height.append(round(i, 2) * 100)

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
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    path = os.path.dirname(__file__) + '/../../be/static/'
    dir = f'{year}{month}/'
    name = generate_random_string()
    filename = f'/{day}_{name}.png'
    directory_path = Path(path + dir)
    try:
        # parents=True 表示递归创建父目录，exist_ok=True 表示如果目录已存在，不会抛出异常
        directory_path.mkdir(parents=True, exist_ok=True) 
    except Exception as error:
        print(f"Error creating directory '{directory_path}': {error}")
    plt.savefig(path + dir + filename)
    
    result = []
    result.append(dir + filename)
    
    result.append("-")
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
