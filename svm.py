from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from time import time
from features_extraction import *
from utils import *
import numpy as np


if __name__ == '__main__':
    # read in images data
    car_images = get_data('vehicles')
    non_car_images = get_data('non-vehicles')
    print('car_sample size:', len(car_images))
    print('non_car_sample size:', len(non_car_images))
    print('all dataset images are read and starting to extract features')

    # extract features from images
    car_features = extract_features(car_images, cspace='YUV', spatial_size=(32,32), hist_bins=32, orient=11, pix_per_cell=16, cell_per_block=2)
    non_car_features = extract_features(non_car_images, cspace='YUV', spatial_size=(32,32), hist_bins=32, orient=11, pix_per_cell=16, cell_per_block=2)

    # normalized features to enhance the robustness
    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_images)), np.zeros(len(non_car_images))))
    scaler = StandardScaler().fit(X)
    normalized_X = scaler.transform(X)

    # split dataset into training and testing sets with randomly shuffles
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.2, shuffle=True)

    # train a linear svm model
    print('finish data preparation, start training...')
    svc = LinearSVC(verbose=10)
    start = time()
    svc.fit(X_train, y_train)
    end = time()
    accuracy = round(svc.score(X_test, y_test),3)
    print('total time used {} seconds, and test accuracy is {}'.format(end-start, accuracy))

    # save model info for later use
    joblib.dump(svc, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
