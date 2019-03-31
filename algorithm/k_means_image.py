import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import k_means as km


IMAGE = mpimg.imread('image.jpg')  # pick up the image as a matrix of pixel
pixels = np.concatenate(IMAGE[:][:])  # all the pixels of the image in an array

EPSILON = 1000  # to test convergence
K_TO_TEST = np.array([1, 2, 3, 4])  # k to test have to be at number of 4 with this script !!

fig = plt.figure(figsize=(8, 8))

print(np.unique(pixels, axis=0).shape[0])  # number of different colors on the image
for i, k in enumerate(K_TO_TEST):
    iteration = 0

    centroids = np.random.rand(k, 3) * 256
    # Initialize a value to keep the last cost value to know when there is a convergence
    last_cost = 0
    cost = EPSILON + 1  # make sure to start the while loop
    # Update the centroid while not convergence
    while not abs(cost - last_cost) < EPSILON:
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
    new_image = new_image.reshape(IMAGE.shape).astype(int)
    # plot it
    fig.add_subplot(2, 2, i + 1)
    plt.imshow(new_image)

plt.show()
