
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import numpy as np
import os
from PIL import Image
import cv2
import joblib
import xgboost as xgb
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier

SIZE = 224
base_dir = "stamp_data"
fake_dir = os.path.join(base_dir, "fake")
real_dir = os.path.join(base_dir, "real")
real_images = []
fake_images = []

def is_image(filename):
    try:
        with Image.open(filename) as img:
            img.verify()
        return True
    except (IOError, OSError) as e:
        return False

for filename in os.listdir(fake_dir)[:100]:
    img_path = os.path.join(fake_dir, filename)
    if is_image(img_path):
        try:
            img = Image.open(img_path).convert('L')
            img = np.array(img)
            img = cv2.resize(img, (SIZE, SIZE))
            fake_images.append(img)
        except Exception as e:
            print(f"Skipping non-image file: {img_path}, Error: {e}")

for filename in os.listdir(real_dir)[:100]:
    img_path = os.path.join(real_dir, filename)
    if is_image(img_path):
        try:
            img = Image.open(img_path).convert('L')
            img = np.array(img)
            img = cv2.resize(img, (SIZE, SIZE))
            real_images.append(img)
        except Exception as e:
            print(f"Skipping non-image file: {img_path}, Error: {e}")




real_images = np.array(real_images)
fake_images = np.array(fake_images)
real_labels = np.zeros((real_images.shape[0], 1))
fake_labels = np.ones((fake_images.shape[0], 1))
images = np.concatenate((real_images, fake_images))
labels = np.concatenate((real_labels, fake_labels))
images = images.reshape(images.shape[0], -1)

train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Apply SMOTE to address class imbalance
smote = SMOTE(random_state=42)
train_data_resampled, train_labels_resampled = smote.fit_resample(train_data_scaled, train_labels)

param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_classifier = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=5)
grid_search_rf.fit(train_data, train_labels.ravel())
best_rf_model = grid_search_rf.best_estimator_
predictions_rf = best_rf_model.predict(test_data)


report_rf = classification_report(test_labels, predictions_rf)
print("Random Forest Classification Report (after optimization):\n", report_rf) # 78%










