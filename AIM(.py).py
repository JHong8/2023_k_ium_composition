#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''

<2023년 제2회 k-ium 의료 인공지능 경진대회>

팀명 : AIM
팀 대표 : 강릉원주대학교 산업공학과 김지희
팀원 : 강릉원주대학교 산업경영공학과 명재홍, 이예승, 정제훈, 최호준

'''


# In[ ]:


'''

팀 대표 이메일 : zlhee@naver.com
팀 대표 연락처 : 010-5141-3362

'''


# In[ ]:


'''

본 프로그램은 'Anaconda'에서 지원하는 'jupyter notebook' 환경에서 작성되었습니다.

플랫폼 : jupyter notebook
GPU : NVIDIA GeForce RTX 3080
GPU API : CUDA 11.2

'''


# In[ ]:


'''

사용한 언어 및 라이브버리 버전
<언어>
1. Python 3.7.16

<라이브러리>
1. numpy
Version: 1.21.6
License: BSD-3-Clause

2. pandas
Version: 1.3.5
License: BSD-3-Clause

3. tensorflow
Version: 2.10.0
License: Apache 2.0

4. sklearn
Version: 1.0.2
License: BSD-3-Clause

5. h5py
Version: 3.8.0
License: BSD 3-Clause

<사전학습 모델>
EfficientnetB0
License: Apache 2.0

'''


# In[ ]:


'''

<Anaconda 가상환경 설정>

# 드라이브 성능에 맞는 GPU API 설치하기
# 본 팀은 CUDA 11.2 와 cudnn 8.1 설치하였습니다.

# 제출한 enviornment.yml파일에는 anaconda 가상환경이 담겨있습니다.
# 따라서 아래의 코드를 cmd에 입력하시면 본 팀의 가상환경을 사용하실 수 있습니다.

conda env create -f environment.yml
conda activate mee
python AIM.py

'''


# In[ ]:


'''

<참고자료>
https://github.com/swotr/tf2-keras-imagenet/blob/master/model.py
https://github.com/qubvel/efficientnet/blob/master/efficientnet/model.py
https://blog.naver.com/PostView.naver?blogId=qwopqwop200&logNo=222053989967&parentCategoryNo=&categoryNo=1&viewDate=&isShowPopularPosts=true&from=search

'''


# In[11]:


# 라이브러리 버전 확인 코드
# 필요시 주석 제거 후 사용

# import numpy as np
# print(np.__version__)

# import pandas as pd
# print(pd.__version__)

# import tensorflow as tf
# print(tf.__version__)

# import sklearn
# print(sklearn.__version__)

# import h5py
# print(h5py.__version__)


# In[1]:


# GPU 확인
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[ ]:


# 뇌동맥류 여부 모델링


# In[49]:


# 환자 Index별 디렉토리 생성 (총 1127개의 디렉토리가 생성됨)

import os
import re
import shutil

# 이미지 데이터가 저장된 폴더 경로
image_folder = "C:/Users/김지희/Desktop/2023_k_ium_composition/train_set/train_set"

# 이미지 데이터를 분류할 폴더 경로
output_folder = 'C:/Users/김지희/Downloads/classified_images'

# 분류된 이미지를 저장할 디렉토리 생성
os.makedirs(output_folder, exist_ok=True)

# 이미지 파일을 분류할 디렉토리를 저장할 딕셔너리
classified_folders = {}

# 이미지 파일을 분류할 디렉토리 생성 및 이미지 파일 복사
for filename in os.listdir(image_folder):
    # 파일 경로
    file_path = os.path.join(image_folder, filename)

    # 파일명에서 숫자 네 자리 추출
    matches = re.findall(r'\d{4}', filename)
    if len(matches) > 0:
        image_index = matches[0]
    else:
        image_index = "others"

    # 분류된 이미지를 저장할 디렉토리 경로
    class_folder = os.path.join(output_folder, image_index)

    # 분류된 이미지를 저장할 디렉토리 생성
    if image_index not in classified_folders:
        os.makedirs(class_folder, exist_ok=True)
        classified_folders[image_index] = class_folder

    # 이미지 파일을 분류된 디렉토리로 복사
    shutil.copy(file_path, class_folder)


# In[50]:


# 분류된 디렉토리의 개수 확인
num_classified_folders = len(classified_folders)

# 분류된 디렉토리의 개수 출력
print("분류된 디렉토리 개수:", num_classified_folders)


# In[51]:


# 뇌동맥류 여부 'Aneurysm'열 데이터 라벨링 및 hdf5 파일 생성


# In[52]:


import os
import numpy as np
from PIL import Image
import pandas as pd
import h5py
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import math


# In[53]:


# 데이터 경로 설정(본인의 파일 경로에 맞게 수정할 것)
csv_file = "C:/Users/김지희/Desktop/2023_k_ium_composition/train_set/train.csv"
image_folder = "C:/Users/김지희/Downloads/classified_images"

# 생성될 hdf5파일 경로 지정
output_hdf5_file = "C:/Users/김지희/Downloads/dataset.hdf5"

# 데이터 프레임 로드
data = pd.read_csv(csv_file)

# 이미지를 저장할 HDF5 파일 생성
with h5py.File(output_hdf5_file, 'w') as hf:
    # 이미지 데이터셋 생성
    image_shape = (256, 256, 8)  # 이미지 크기와 이미지 개수
    image_dtype = np.uint8  # 이미지 데이터 타입
    image_dataset = hf.create_dataset('images', (len(data), *image_shape), dtype=image_dtype)
    
    # 라벨 데이터셋 생성
    label_dtype = np.uint8  # 라벨 데이터 타입
    label_dataset = hf.create_dataset('labels', (len(data),), dtype=label_dtype)
    
    # 이미지 데이터와 라벨링
    for index, row in data.iterrows():
        image_index = str(row['Index'])
        class_folder = os.path.join(image_folder, image_index)
        
        # 폴더가 존재하는 경우
        if os.path.isdir(class_folder):
            # 이미지 파일 목록
            image_files = os.listdir(class_folder)
            
            # 이미지 파일이 8장인 경우
            if len(image_files) == 8:
                # 이미지 로드 및 전처리
                images = []
                for filename in image_files:
                    file_path = os.path.join(class_folder, filename)
                    img = Image.open(file_path)
                    img = img.resize(image_shape[:2])  # 이미지 크기만 조정
                    img_array = np.array(img, dtype=image_dtype)[:,:,0]
                    images.append(img_array)
                
                # 이미지를 깊이(dimension)로 쌓아 3차원 데이터로 변환
                image_data = np.stack(images, axis=-1)
                
                # HDF5 데이터셋에 저장
                image_dataset[index] = image_data
                label_dataset[index] = int(row['Aneurysm'])


# In[54]:


# efficientnet 모델 생성


# In[55]:


input_channels = 8
se_ratio = 4
expand_ratio = 6
width_coefficient = 1.0
depth_coefficient = 1.0
default_resolution = 256
depth_divisor= 8 
dropout_rate = 0.2
drop_connect_rate = 0.2
kernel_size = [3,3,5,3,5,5,3]
num_repeat = [1,2,2,3,3,4,1]
output_filters = [16,24,40,80,112,192,320]
strides = [1,2,2,2,1,2,1]
MBConvBlock_1_True  =  [True,False,False,False,False,False,False]


# In[56]:


def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))


# In[57]:


def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


# In[58]:


class DropConnect(layers.Layer):
    def __init__(self, drop_connect_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training):
        def _drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += K.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = tf.math.divide(inputs, keep_prob) * binary_tensor
            return output

        return K.in_train_phase(_drop_connect, inputs, training=training)


# In[59]:


def SEBlock(filters,reduced_filters):
    def _block(inputs):
        x = layers.GlobalAveragePooling2D()(inputs)
        x = layers.Reshape((1,1,x.shape[1]))(x)
        x = layers.Conv2D(reduced_filters, 1, 1)(x)
        x = tf.keras.activations.gelu(x)
        x = layers.Conv2D(filters, 1, 1)(x)
        x = layers.Activation('sigmoid')(x)
        x = layers.Multiply()([x, inputs])
        return x
    return _block


# In[60]:


def MBConvBlock(x, kernel_size, strides, drop_connect_rate, output_channels, MBConvBlock_1_True=False):
    output_channels = round_filters(output_channels, width_coefficient, depth_divisor)
    if MBConvBlock_1_True:
        block = layers.DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False)(x)
        block = layers.BatchNormalization()(block)
        block = tf.keras.activations.gelu(block)
        block = SEBlock(x.shape[3], x.shape[3] // se_ratio)(block)
        block = layers.Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(block)
        block = layers.BatchNormalization()(block)
        return block

    channels = x.shape[3]
    expand_channels = channels * expand_ratio
    block = layers.Conv2D(expand_channels, (1, 1), padding='same', use_bias=False)(x)
    block = layers.BatchNormalization()(block)
    block = tf.keras.activations.gelu(block)
    block = layers.DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False)(block)
    block = layers.BatchNormalization()(block)
    block = tf.keras.activations.gelu(block)
    block = SEBlock(expand_channels, channels // se_ratio)(block)
    block = layers.Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(block)
    block = layers.BatchNormalization()(block)
    if x.shape[3] == output_channels:
        block = DropConnect(drop_connect_rate)(block)
        block = layers.Add()([block, x])
    return block


# In[61]:


def EffNet(num_classes):
    x_input = layers.Input(shape=(default_resolution, default_resolution, input_channels))    
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), (3, 3), 2, padding='same', use_bias=False)(x_input)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.gelu(x)
    num_blocks_total = sum(num_repeat)
    block_num = 0
    for i in range(len(kernel_size)):
        round_num_repeat = round_repeats(num_repeat[i], depth_coefficient)
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = MBConvBlock(x, kernel_size[i], strides[i], drop_rate, output_filters[i], MBConvBlock_1_True=MBConvBlock_1_True[i])
        block_num += 1
        if round_num_repeat > 1:
            for bidx in range(round_num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                x = MBConvBlock(x, kernel_size[i], 1, drop_rate, output_filters[i], MBConvBlock_1_True=MBConvBlock_1_True[i])
                block_num += 1
    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.gelu(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x)
    return model


# In[62]:


from sklearn.model_selection import train_test_split
import h5py
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# 데이터 로드
with h5py.File("C:/Users/김지희/Downloads/dataset.hdf5", 'r') as hdf:
    images = np.array(hdf['images'])
    labels = np.array(hdf['labels'])

# 정규화
images = images / 255.0

# 레이블 형상 변경
labels = np.reshape(labels, (labels.shape[0], 1))
labels = np.expand_dims(labels, axis=-1)

# EfficientNet 모델 빌드
model = EffNet(num_classes=2)

# Adam 옵티마이저의 학습률 낮추기
optimizer = Adam(lr=0.0001)

# 모델 컴파일
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 데이터 학습
model.fit(images, labels, batch_size=8, epochs=10, validation_split=0.3, verbose=1)

# 예측 수행
predictions = model.predict(images)

# 예측 결과 출력
print(predictions)


# In[63]:


# 예측 결과 데이터 프레임으로 변환
data_df = pd.DataFrame(predictions)
data_df


# In[64]:


# test.csv에는 50개의 데이터만 들어있으므로 예측 결과 중 50개만 출력
a = data_df.iloc[:50, :]


# In[65]:


# 0으로 코딩되어있는 test.csv 불러오기
df = pd.read_csv("C:/Users/김지희/Desktop/2023_k_ium_composition/test_set/test_set/test.csv")


# In[66]:


# test.csv의 Aneurysm열에 예측 결과 값 입력
a = list(a[0])
df['Aneurysm'] = a


# In[68]:


# 데이터가 잘 들어갔는지 확인
df


# In[69]:


# Aneurysm(뇌동맥류 여부) 예측 csv 생성
df.to_csv('C:/Users/김지희/Downloads/output_Aneurysm.csv', index=False)


# In[70]:


# 뇌동맥류 위치 정보 모델링


# In[71]:


# 뇌동맥류 위치 정보를 가진 21개의 열 각각의 데이터 라벨링 및 hdf5 파일 생성


# In[72]:


import os
import h5py
import pandas as pd
import numpy as np
from PIL import Image

# 본인 컴퓨터에 저장되어있는 train.csv 파일 경로 수정해서 넣기
csv_file = "C:/Users/김지희/Desktop/2023_k_ium_composition/train_set/train.csv"

# 본인 컴퓨터에 저장되어있는 classified_images 파일 경로 수정해서 넣기
image_folder = "C:/Users/김지희/Downloads/classified_images"

# 출력되는 hdf5 파일 본인 컴퓨터에 저장할 위치 수정해서 넣기
output_folder = "C:/Users/김지희/Downloads/version2"

# train.csv에 있는 뇌동맥류 위치 정보 변수
location_columns = ['L_ICA', 'R_ICA', 'L_PCOM', 'R_PCOM', 'L_AntChor', 'R_AntChor',
                    'L_ACA', 'R_ACA', 'L_ACOM', 'R_ACOM', 'L_MCA', 'R_MCA',
                    'L_VA', 'R_VA', 'L_PICA', 'R_PICA', 'L_SCA', 'R_SCA', 'BA',
                    'L_PCA', 'R_PCA']

# csv 데이터 로드
data = pd.read_csv(csv_file)

# 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 21개의 hdf5 파일 생성 반복문
for col in location_columns:
    # 파일명 생성
    output_file = os.path.join(output_folder, col + '.hdf5')

    # HDF5 파일 생성
    with h5py.File(output_file, 'w') as hf:
        # 이미지 데이터셋 생성
        image_shape = (256, 256, 8)  # 이미지 크기와 이미지 개수
        image_dtype = np.uint8  # 이미지 데이터 타입
        image_dataset = hf.create_dataset('images', (len(data), *image_shape), dtype=image_dtype)

        # 라벨 데이터셋 생성
        label_dtype = np.uint8  # 라벨 데이터 타입
        label_dataset = hf.create_dataset('labels', (len(data), 1), dtype=label_dtype)

        # 이미지 데이터와 라벨링
        for index, row in data.iterrows():
            image_index = str(row['Index'])
            class_folder = os.path.join(image_folder, image_index)

            # 폴더가 존재하는 경우
            if os.path.isdir(class_folder):
                # 이미지 파일 목록
                image_files = os.listdir(class_folder)

                # 이미지 파일이 8장인 경우
                if len(image_files) == 8:
                    # 이미지 로드 및 전처리
                    images = []
                    for filename in image_files:
                        file_path = os.path.join(class_folder, filename)
                        img = Image.open(file_path)
                        img = img.resize(image_shape[:2])  # 이미지 크기만 조정
                        img_array = np.array(img, dtype=image_dtype)[:, :, 0]
                        images.append(img_array)

                    # 이미지를 깊이(dimension)로 쌓아 3차원 데이터로 변환
                    image_data = np.stack(images, axis=-1)

                    # HDF5 데이터셋에 저장
                    image_dataset[index] = image_data
                    label_dataset[index] = row[col].astype(label_dtype)


# In[73]:


# efficientnet 모델 생성


# In[74]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import math


# In[75]:


input_channels = 8
se_ratio = 4
expand_ratio = 6
width_coefficient = 1.0
depth_coefficient = 1.0
default_resolution = 256
depth_divisor= 8 
dropout_rate = 0.2
drop_connect_rate = 0.2
kernel_size = [3,3,5,3,5,5,3]
num_repeat = [1,2,2,3,3,4,1]
output_filters = [16,24,40,80,112,192,320]
strides = [1,2,2,2,1,2,1]
MBConvBlock_1_True  =  [True,False,False,False,False,False,False]


# In[76]:


def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))


# In[77]:


def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


# In[78]:


class DropConnect(layers.Layer):
    def __init__(self, drop_connect_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training):
        def _drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += K.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = tf.math.divide(inputs, keep_prob) * binary_tensor
            return output

        return K.in_train_phase(_drop_connect, inputs, training=training)


# In[79]:


def SEBlock(filters,reduced_filters):
    def _block(inputs):
        x = layers.GlobalAveragePooling2D()(inputs)
        x = layers.Reshape((1,1,x.shape[1]))(x)
        x = layers.Conv2D(reduced_filters, 1, 1)(x)
        x = tf.keras.activations.gelu(x)
        x = layers.Conv2D(filters, 1, 1)(x)
        x = layers.Activation('sigmoid')(x)
        x = layers.Multiply()([x, inputs])
        return x
    return _block


# In[80]:


def MBConvBlock(x, kernel_size, strides, drop_connect_rate, output_channels, MBConvBlock_1_True=False):
    output_channels = round_filters(output_channels, width_coefficient, depth_divisor)
    if MBConvBlock_1_True:
        block = layers.DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False)(x)
        block = layers.BatchNormalization()(block)
        block = tf.keras.activations.gelu(block)
        block = SEBlock(x.shape[3], x.shape[3] // se_ratio)(block)
        block = layers.Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(block)
        block = layers.BatchNormalization()(block)
        return block

    channels = x.shape[3]
    expand_channels = channels * expand_ratio
    block = layers.Conv2D(expand_channels, (1, 1), padding='same', use_bias=False)(x)
    block = layers.BatchNormalization()(block)
    block = tf.keras.activations.gelu(block)
    block = layers.DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False)(block)
    block = layers.BatchNormalization()(block)
    block = tf.keras.activations.gelu(block)
    block = SEBlock(expand_channels, channels // se_ratio)(block)
    block = layers.Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(block)
    block = layers.BatchNormalization()(block)
    if x.shape[3] == output_channels:
        block = DropConnect(drop_connect_rate)(block)
        block = layers.Add()([block, x])
    return block


# In[81]:


def EffNet(num_classes):
    x_input = layers.Input(shape=(default_resolution, default_resolution, input_channels))    
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), (3, 3), 2, padding='same', use_bias=False)(x_input)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.gelu(x)
    num_blocks_total = sum(num_repeat)
    block_num = 0
    for i in range(len(kernel_size)):
        round_num_repeat = round_repeats(num_repeat[i], depth_coefficient)
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = MBConvBlock(x, kernel_size[i], strides[i], drop_rate, output_filters[i], MBConvBlock_1_True=MBConvBlock_1_True[i])
        block_num += 1
        if round_num_repeat > 1:
            for bidx in range(round_num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                x = MBConvBlock(x, kernel_size[i], 1, drop_rate, output_filters[i], MBConvBlock_1_True=MBConvBlock_1_True[i])
                block_num += 1
    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.gelu(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x)
    return model


# In[82]:


# efficientnet 모델 학습


# In[83]:


import pandas as pd
import h5py
import numpy as np
from tensorflow.keras.optimizers import Adam

# train.csv에 있는 뇌동맥류 위치 정보 변수
location_columns = ['L_ICA', 'R_ICA', 'L_PCOM', 'R_PCOM', 'L_AntChor', 'R_AntChor',
                    'L_ACA', 'R_ACA', 'L_ACOM', 'R_ACOM', 'L_MCA', 'R_MCA',
                    'L_VA', 'R_VA', 'L_PICA', 'R_PICA', 'L_SCA', 'R_SCA', 'BA',
                    'L_PCA', 'R_PCA']

# 뇌동맥류 위치 정보 업데이트
data = pd.DataFrame(columns=location_columns)

def threshold_predictions(predictions, threshold):
    # 임계값을 기준으로 예측값을 0 또는 1로 변환
    return np.where(predictions > threshold, 1, 0)

thresholds = {}

# 반복문을 통해 각 hdf5 파일에 대해 실행
for location_column in location_columns:
    hdf5_file = f"C:/Users/김지희/Downloads/version2/{location_column}.hdf5"

    with h5py.File(hdf5_file, 'r') as hdf:
        images = np.array(hdf['images'])
        labels = np.array(hdf['labels'])

    # 정규화
    images = images / 255.0

    # 레이블 형상 변경
    labels = np.reshape(labels, (labels.shape[0], 1))
    labels = np.expand_dims(labels, axis=-1)

    # EfficientNet 모델 빌드
    model = EffNet(num_classes=2)

    # Adam 옵티마이저의 학습률 낮추기
    optimizer = Adam(lr=0.0001)

    # 모델 컴파일
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 데이터 학습
    model.fit(images, labels, batch_size=8, epochs=7, validation_split=0.3, verbose=1)

    # 예측 수행
    predictions = model.predict(images)

    # 예측 결과 출력
    print(predictions)

    # 예측 값들의 평균과 표준편차 계산
    predictions_mean = np.mean(predictions)
    predictions_std = np.std(predictions)

    # 각 변수에 해당하는 평균과 표준편차를 기준으로 임계값 설정
    threshold = predictions_mean + predictions_std

    # 예측값을 임계값에 따라 0 또는 1로 변환
    thresholded_predictions = threshold_predictions(predictions, threshold)

    # 데이터프레임에 업데이트
    data[location_column] = thresholded_predictions.flatten()

# 결과 출력
print(data)


# In[84]:


data['R_AntChor'].value_counts()


# In[85]:


# 위에서 생성했던 Aneurysm(뇌동맥류 여부) 예측 csv 불러오기
df = pd.read_csv('C:/Users/김지희/Downloads/output_Aneurysm.csv')


# In[86]:


# 뇌동맥류 위치 정보 업데이트한 데이터 프레임에서 데이터 50개만 추출
aa = data.iloc[:50, :]


# In[87]:


# 데이터프레임의 데이터를 리스트 형태로 변환
values = aa.iloc[:, :].values.tolist()


# In[88]:


# 새로운 데이터프레임 생성 및 값 넣기
df.iloc[:, 2:] = values
df


# In[89]:


# 최종 출력 파일 생성
df.to_csv('C:/Users/김지희/Downloads/output.csv', index=False)


# In[ ]:




