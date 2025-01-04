SIZE = 224
train_dir = "/home/marine/PycharmProjects/pythonProject/signature_data/data"
real_images, forged_images = [], []

for per in os.listdir(train_dir):
    for data in glob.glob(os.path.join(train_dir, per, '*.*')):
        img = Image.open(data).convert('L')  # Convert to grayscale
        img = np.array(img)
        img = cv2.resize(img, (SIZE, SIZE))
        if per[-1] == 'g':
            forged_images.append(img)
        else:
            real_images.append(img)

real_images = np.array(real_images)
forged_images = np.array(forged_images)
real_labels = np.zeros((real_images.shape[0], 1))
forged_labels = np.ones((forged_images.shape[0], 1))
images = np.concatenate((real_images, forged_images))
labels = np.concatenate((real_labels, forged_labels))
images = images.reshape(images.shape[0], -1)

train_data, test_data, train_labels, test_labels = (train_test_split(
                        images, labels, test_size=0.2, random_state=42))

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
