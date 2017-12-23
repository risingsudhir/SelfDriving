import pickle
import numpy as np
import pandas as pd
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt


TRAIN_FILE = './data/train.p'
VALID_FILE = './data/valid.p'
TEST_FILE  = './data/test.p'

PREPROCESS_FILE  = './data/preprocess_{}.p'
SIGN_NAMES_FILE  = './data/signnames.csv'
NEW_IMAGE_PATH   = './data/new-images/'
NEW_IMAGE_LABELS = './data/new-images/labels.csv'


def _read_data(filename):
    """
    Read the data stored in pickle format
    """
    with open(filename, mode = 'rb') as file: 
        data = pickle.load(file)
    
    return data
        

def get_sign_names():
    """
    Read sign names from file
    : return: dictionary with label id and name
    """
    
    label_names = pd.read_csv(SIGN_NAMES_FILE)
    
    return { id:name for id, name in zip(label_names.ClassId, label_names.SignName)}


def get_train_validation_test_data():
    """
    Get the training, validation and test data    
    """

    train = _read_data(TRAIN_FILE)
    valid = _read_data(VALID_FILE)
    test  = _read_data(TEST_FILE)
    
    return (train, valid, test)
    

def print_data_stats(X, Y):
    """
    Print stats of the feature and label data set
    """
        
    # get sign id and name map
    sign_names    = get_sign_names()
    total_samples = len(X)
    samples_count = {sign_names[i]:j for i,j in zip(*np.unique(Y, return_counts=True))}
    
    print('Samples: {:,}'.format(total_samples))   
    print('Classes: {:}'.format(len(np.unique(Y))))
    
    for id in sign_names.keys():
        print("ID: {:2}, Samples: {:4}, Name: {}".format(id, samples_count.get(sign_names[id], 0), sign_names[id]))
        
    print('Distribution of Classes \r\n')
        
    plt.hist(list(samples_count.values()), bins = len(sign_names))
            
        
def show_sample(X, Y, sample_id):
    """
    Show selected sample image from data set
    """
    
    # validate sample id
    if not (0 <= sample_id < len(X)):
        print('{} samples only.  {:,} is out of range.'.format(len(X), sample_id))
        return None
    
    # get sign id and name map
    sign_names    = get_sign_names()
    sample_image = X[sample_id]
    sample_label = Y[sample_id]
    
    print('\nSample Image {}:'.format(sample_id))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Id: {} Name: {}'.format(sample_label, sign_names[sample_label]))
    
    plt.axis('off')
    plt.imshow(sample_image)

    
def show_image(image, channel = None):
    """
    Display given image
    """    
    
    plt.axis('off')
    
    if channel:
        plt.imshow(image, cmap = channel)
    else:
        plt.imshow(image)
    
    
def show_images(images, labels, channel = None):
    """
    Display given images in a single cell
    """    
    count = len(images)
    fig = plt.figure()
    
    for i in range(count):    
        plot = fig.add_subplot(1, count, (i + 1))
        plot.set_title(labels[i])
        plt.imshow(images[i])
    
    plt.show()
    
    sign_names = get_sign_names()
    
    for id in labels:
        print("ID: {:2}, Name: {}".format(id, sign_names[id]))
       
               
def preprocess_save_data(X, Y, preprocessor, data_type):
    """
    Preprocess features and labels data using proprocessor 
    and save to the disk
    : X: feature data set
    : Y: label data set
    : preprocessor: method to transform the data
    : data_type: suffix to identity the data type
    """
    
    trans_X, trans_Y = preprocessor(X, Y)
    
    file_name = PREPROCESS_FILE.format(data_type)
        
    pickle.dump((trans_X, trans_Y), open(file_name, 'wb'))
    
    print('Saved pre-processed {} data to file {}'.format(data_type, file_name))
            

def get_preprocess_data(data_type):
    """
    Load the pre-processed features and labels data set from the disk
    : data_type: suffix to identity the data type (train, valid, test)
    """
    
    file_name = PREPROCESS_FILE.format(data_type) 
    
    return _read_data(file_name)
           
        
def batch_features_labels(X, Y, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        yield X[start:end], Y[start:end]
        
        
def get_new_test_images():
    """
    Get the new set of test images not part of training, validation and test data set.
    """
    
    files = [file for file in os.listdir(NEW_IMAGE_PATH) if file.endswith('.jpg')]
    count = len(files)
    
    images = np.ndarray(shape = (count, 32, 32, 3), dtype = np.uint8)
    labels = np.array(range(count))
                
    labels_data = pd.read_csv(NEW_IMAGE_LABELS)
    labels_map = { name:id for id, name in zip(labels_data.ClassId, labels_data.ImageName)}
                      
    for i in range(count):
        images[i] = Image.open(NEW_IMAGE_PATH + files[i])
        labels[i] = labels_map[files[i]]
        
    return (images, labels)
    
    
def show_top_n_predictions(images, labels, top_n_predictions):
    """
    Show top n predictions for images
    : images: image vector to show
    : labels: actual labels of images
    : top_n_predictions: top n predictions by neural network of image vector
    """
    count = len(images)
    sign_names = get_sign_names()
    
    for i in range(count):
        fig = plt.figure()
        plot = fig.add_subplot(1, 1, 1)
        plot.set_title(labels[i])
        plt.imshow(images[i])
        
        plt.show()
        
        predictions = top_n_predictions[i]
        
        for j in range(len(predictions)):
            print("Top-{}: ID: {:2}, Name: {}".format(j+1, predictions[j], sign_names[predictions[j]]))
        
        print("IMAGE: ID: {:2}, Name: {} \r\n".format(labels[i], sign_names[labels[i]]))