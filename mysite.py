import streamlit as st 
import tensorflow as tf 
import efficientnet.tfkeras as efn 
from PIL import Image
import numpy as np
import io

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB6]

AUTO = tf.data.experimental.AUTOTUNE

st.set_page_config(
    page_title="Drought-watch",  # default page title
    layout="centered"
)

def build_model(dim=224, ef=0):
    inp = tf.keras.layers.Input(shape=(dim,dim,10))
    base = EFNS[ef](input_shape=(dim,dim,10),weights=None,include_top=False) #Change imagnet to noisy-student here
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=0)
    model.compile(optimizer=opt,loss=loss_bce)
    return model

@st.cache
def modelCache():
    model = build_model(dim=65, ef=0)
    model.load_weights('input/droughtwatch.hdf5')
    return model

def clean_rgb(x):
    st.warning("You have inputted a 3 channel image. This model was trained on 10 channel images, please use 10 channel images for the best accuracy.")
    x = x.convert('RGB')
    x = x.resize((65, 65))
    x = np.asarray(x)
    x = np.dstack((x, np.zeros((65, 65, 7)))) # expand to 10 channels
    x = np.expand_dims(x, axis=0)/255.
    return x 

keylist = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']
def read_labeled_tfrecord(example):
    tfrec_format = {
    'B1': tf.io.FixedLenFeature([], tf.string),
    'B2': tf.io.FixedLenFeature([], tf.string),
    'B3': tf.io.FixedLenFeature([], tf.string),
    'B4': tf.io.FixedLenFeature([], tf.string),
    'B5': tf.io.FixedLenFeature([], tf.string),
    'B6': tf.io.FixedLenFeature([], tf.string),
    'B7': tf.io.FixedLenFeature([], tf.string),
    'B8': tf.io.FixedLenFeature([], tf.string),
    'B9': tf.io.FixedLenFeature([], tf.string),
    'B10': tf.io.FixedLenFeature([], tf.string),
    'B11': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    }              
    example = tf.io.parse_single_example(example, tfrec_format)
    return [example[i] for i in keylist], tf.cast(example['label'], tf.float32) # required float32 for tpu, CHANGE CLIP

def prepare_image(img, augment=True, dim=65):    
    img = tf.io.decode_raw(img, out_type=tf.uint8)
    img = tf.cast(img, tf.float32) / 255.0 # CHEC
    
    if augment:
        img = transform(img,DIM=dim)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
            
    return tf.cast(tf.reshape(img, [dim,dim,10]), tf.float32)

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
        for filename in filenames]
    return np.sum(n)

def get_dataset(files, augment = False, shuffle = False, repeat = False, 
                labeled=True, return_image_names=True, batch_size=16, dim=256):
    
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    
    if repeat:
        ds = ds.repeat()

    if shuffle: 
        ds = ds.shuffle(batch_size)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else: 
        ds = ds.map(read_unlabelled_tfrecord, num_parallel_calls=AUTO)
    
    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, dim=dim), tf.one_hot(imgname_or_label, 4, dtype=tf.float32), [4]), num_parallel_calls=AUTO)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return ds

def clean_tfrec(x):
    dataset = get_dataset(x)
    return dataset

def predict(x):
    model = modelCache()
    label = model.predict(x)
    maxlabel = np.argmax(label)
    prob = label[0][maxlabel]
    prob = '%.2f' % (prob*100)
    cows = maxlabel if maxlabel!=3 else "3+"
    st.success(f"Land is able to support {cows} cows per square km. Certainty of {prob}%")
    return prob, label

st.markdown("<style> .reportview-container .main footer {visibility: hidden;}    #MainMenu {visibility: hidden;}</style>", unsafe_allow_html=True)

st.write('<h1 style="font-weight:400; color:red">DroughtWatch</h1>', unsafe_allow_html=True)
st.write('### Predict foliage cover of a 2km square 10 channel (or 3 channel) image with %s%% '% 77 + 'accuracy')
st.write('This model is superior to a currently deployed model in Kenya. Please see hackathon submission down below. Created in 12 hours during HackCamp 2020')

st.write('Feel free to [download images](https://github.com/stanleyjzheng/drought-watch/tree/master/example_images) to test. There are two file types: conventional images, which are 3 channel (RGB), and npy, which contain 10 channel images resulting in greater accuracy.')
st.write('Convert from tfrecord to npy with our GitHub repo linked below.')

userFile = st.file_uploader('Please upload an image or tfrecord', type=['jpg', 'jpeg', 'png', 'npy'])
if userFile is not None:
    try:
        img = Image.open(userFile)
        st.image(img, use_column_width = True, caption = 'Uploaded image')
        img = clean_rgb(img)
    except:
        img = userFile.getvalue()
        img = np.load(io.BytesIO(img))
        img = np.reshape(img, (65, 65, 10))
        img = np.expand_dims(img, axis=0)
    with st.spinner(text = 'Loading...'):
        label = predict(img)

st.markdown('''
[Source Code](https://github.com/stanleyjzheng/drought-watch)

[My GitHub with other projects](https://github.com/stanleyjzheng)

[Hackathon submission](https://devpost.com)''')