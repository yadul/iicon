from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(firstval, secondval):

    err = np.sum((firstval.astype("float") - secondval.astype("float")) ** 2)
    err /= float(firstval.shape[0] * firstval.shape[1])
    
    return err

def compare_images(firstval, secondval, title):
    
    m = mse(firstval, secondval)
    s = ssim(firstval, secondval)
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))


    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(firstval, cmap = plt.cm.gray)
    plt.axis("off")


    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(secondval, cmap = plt.cm.gray)
    plt.axis("off")

    plt.show()

hd = cv2.imread("images/car_hd.jpeg")
lt = cv2.imread("images/car_lt.jpeg")


hd = cv2.cvtColor(hd, cv2.COLOR_BGR2GRAY)
lt = cv2.cvtColor(lt, cv2.COLOR_BGR2GRAY)


fig = plt.figure("Images")
images = ("hd", hd), ("lt", lt)

for (i, (name, image)) in enumerate(images):
    
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap = plt.cm.gray)
    plt.axis("off")
