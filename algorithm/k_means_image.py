import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import k_means as km


image = mpimg.imread('image.jpg')  # pick up the image as a matrix of pixel
pixels = np.concatenate(image[:][:])  # all the pixels of the image in an array

epsilon = 1000  # to test convergence
k_to_test = np.array([2, 4, 8, 16])  # k to test have to be at number of 4 with this script !!

fig = plt.figure(figsize=(8, 8))

print(np.unique(pixels, axis=0).shape[0])  # number of different colors on the image
for i, k in enumerate(k_to_test):
    iteration = 0

    centroids = np.random.rand(k, 3) * 256
    # Initialize a value to keep the last cost value to know when there is a convergence
    last_cost = 0
    cost = epsilon + 1  # make sure to start the while loop
    # Update the centroid while not convergence
    while not abs(cost - last_cost) < epsilon:  # np.array_equal(centroids, last_centroids):
        print('iteration  {0}... '.format(iteration + 1))
        # keep the current cost before the adjustments to know if there is a convergence
        last_cost = cost
        # pick up the nearest centroid indexes of each samples
        nearest = km.nearest_centroid(pixels, centroids)
        # adjust the current centroids
        k_centroids = km.adjust_centroid(pixels, nearest, centroids)
        # calculation of the current cost
        cost = km.calculate_cost(pixels, centroids)
        print('cost : {0}'.format(cost))
        print('centroids : \n{0}'.format(centroids))
        iteration += 1
    # reassociate each color to its nearest centroid
    nearest = km.nearest_centroid(pixels, centroids)
    # create the pixel array
    new_image = np.array([centroids[number] for _, number in enumerate(nearest)])
    # reform the image
    new_image = new_image.reshape(image.shape).astype(int)
    # plot it
    fig.add_subplot(2, 2, i + 1)
    plt.imshow(new_image)

plt.show()
