from flask import Flask, request, jsonify, send_from_directory, render_template, flash, redirect, url_for,  session
from werkzeug.utils import secure_filename
import os
import numpy as np
import pickle
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
from keras._tf_keras.keras.applications.resnet50 import ResNet50
from keras._tf_keras.keras.preprocessing import image
from PIL import Image  
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import mysql.connector

from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

BASE_URL = "http://127.0.0.1:5000"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.path.join(APP_ROOT, 'uploads')
DATA = os.path.join(APP_ROOT, 'data')
app.config['UPLOAD'] = UPLOAD
app.config['DATA'] = DATA


#connect database
def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="food_img_search"
    )
    return conn


def get_feature_vectors():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """SELECT id, feature FROM food_imgs"""
    cursor.execute(query)
    rows = cursor.fetchall()
    
    ids = []
    feature_vectors = []
    for row in rows:
        id = row[0]
        ids.append(id)
        feature_blob = row[1]
        feature_vector = pickle.loads(feature_blob)
        feature_vectors.append(np.array(feature_vector))
    
    conn.close()
    return ids, feature_vectors

ids, features = get_feature_vectors()

# Kích thước ảnh đầu vào 
img_width, img_height = 224,224

# Khởi tạo mô hình ResNet50 và bỏ đi lớp fully connected (top layer)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Chuyển đổi thành mảng numpy
X_train = np.array(features)

# Trích xuất đặc trưng từ mô hình ResNet50
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

def get_food_info(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query_select = """
    SELECT food_name, description, img_path FROM food_imgs
    WHERE id = %s
    """
    cursor.execute(query_select, (id,))
    food_data = cursor.fetchone()
    conn.close()
    
    if food_data:
        return food_data[0], food_data[1], food_data[2]
    else:
        return "", "", ""

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD'], filename)

@app.route('/data/<folder>/<filename>')
def data_image(folder,filename):
    return send_from_directory(app.config['DATA'], folder + '/' +filename)

@app.route('/', methods=['GET'])
def index():
    return render_template('test.html')


@app.route('/', methods=['POST'])
def recognize():
    file = request.files.get('image')
    if file:
        filename = secure_filename(file.filename)
        file_name_random = get_random_string(12) + filename
        filepath = os.path.join(app.config['UPLOAD'], file_name_random)
        file.save(filepath)

        # Khi có một ảnh mới đầu vào
        new_image_path = filepath

        # Trích xuất đặc trưng của ảnh mới
        new_image_feature = extract_features(new_image_path)

        x_test = [new_image_feature]

        # Tính toán cosine similarity giữa vector đặc trưng của hình ảnh mới và tất cả các hình ảnh trong X_train
        cos_similarities = cosine_similarity(x_test, X_train)

        # Sắp xếp các món ăn dựa trên cosine similarity
        sorted_indices = np.argsort(cos_similarities[0])[::-1]

        # Lấy danh sách các món ăn tương tự 
        num_similar_items = 40
        similar_items = sorted_indices[:num_similar_items]

        list_image_urls = []
        list_food_names = []
        descriptions = []
        list_images = []

        # Duyệt qua từng đường dẫn ảnh trong danh sách similar_items
        for i in similar_items:
            food_id = ids[i]
            food_name, description, img_path = get_food_info(food_id)

            list_food_names.append(food_name.title())
            descriptions.append(description if description else "")
            list_images.append(img_path)

            if 'data' in img_path:
                imageUrl = 'http://' + BASE_URL + '/data' + img_path.split('data')[1].replace("\\", "/")
                list_image_urls.append(imageUrl)
            else:
                imageUrl = 'http://' + BASE_URL + '/data' + img_path.replace("\\", "/")
                

        return jsonify({
            'list_images': list_images,
            'list_image_urls': list_image_urls,
            'list_food_names': list_food_names,
            'descriptions': descriptions
        })
    else:
        return jsonify({'message': 'No file uploaded!'})


if __name__ == "__main__":
    app.run(debug=True)

