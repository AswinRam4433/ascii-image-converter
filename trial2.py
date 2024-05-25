# import cv2
# im_gray = cv2.imread(r"D:\LangChain\Conan Images 512x512\resized-image-Promo.jpeg", cv2.IMREAD_GRAYSCALE)
# (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# thresh = 127
# im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
# cv2.imwrite('bw_image.png', im_bw)


# import the necessary packages
# from sklearn.cluster import MiniBatchKMeans
# import numpy as np
# import argparse
# import cv2

# cluster_n = int(input("Enter the number of clusters you want to create "))
# image = cv2.imread(r"D:\LangChain\Conan Images 512x512\resized-image-Promo.jpeg")
# (h, w) = image.shape[:2]
# image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# # reshape the image into a feature vector so that k-means can be applied
# image = image.reshape((image.shape[0] * image.shape[1], 3))

# # apply k-means using the specified number of clusters and then create the quantized image based on the predictions
# clt = MiniBatchKMeans(n_clusters=cluster_n)
# labels = clt.fit_predict(image)

# # Print the cluster labels for each pixel
# # for i, label in enumerate(labels):
# #     print(f"Pixel {i} belongs to cluster {label}")

# quant = clt.cluster_centers_.astype("uint8")[labels]

# # reshape the feature vectors to images
# quant = quant.reshape((h, w, 3))
# image = image.reshape((h, w, 3))

# # convert from L*a*b* to RGB
# quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
# image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

# # display the images and wait for a keypress
# cv2.imshow("image", np.hstack([image, quant]))
# cv2.waitKey(0)



# from sklearn.cluster import MiniBatchKMeans
# import numpy as np
# import argparse
# import cv2

# cluster_n = int(input("Enter the number of clusters you want to create "))
# image = cv2.imread(r"D:\LangChain\Conan Images 512x512\resized-image-Promo.jpeg")
# (h, w) = image.shape[:2]
# image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# # reshape the image into a feature vector so that k-means can be applied
# image_flat = image.reshape((image.shape[0] * image.shape[1], 3))

# # apply k-means using the specified number of clusters and then create the quantized image based on the predictions
# clt = MiniBatchKMeans(n_clusters=cluster_n)
# labels = clt.fit_predict(image_flat)

# # Create an image where each pixel is represented by its cluster label
# labels_image = labels.reshape((h, w))

# # Convert label image to uint8 for visualization
# labels_image = labels_image.astype(np.uint8)

# # Display the labels image
# cv2.imshow("Labels Image", labels_image)
# cv2.waitKey(0)


# from sklearn.cluster import MiniBatchKMeans
# import numpy as np
# import argparse
# import cv2

# def label_to_alphabet(label):
#     return chr(ord('a') + label)

# cluster_n = int(input("Enter the number of clusters you want to create "))
# image = cv2.imread(r"D:\LangChain\Conan Images 512x512\resized-image-Promo.jpeg")
# (h, w) = image.shape[:2]
# image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# # reshape the image into a feature vector so that k-means can be applied
# image_flat = image.reshape((image.shape[0] * image.shape[1], 3))

# # apply k-means using the specified number of clusters and then create the quantized image based on the predictions
# clt = MiniBatchKMeans(n_clusters=cluster_n)
# labels = clt.fit_predict(image_flat)

# # Convert numeric labels to alphabets
# labels_alphabet = [label_to_alphabet(label) for label in labels]

# # Encode alphabet characters to numerical values
# labels_numeric = [ord(char) for char in labels_alphabet]

# # Create an image where each pixel is represented by its cluster label
# labels_image = np.array(labels_numeric).reshape((h, w)).astype(np.uint8)  # Convert to np.uint8

# # Display the labels image
# cv2.imshow("Labels Image", labels_image)
# cv2.waitKey(0)



from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2

def label_to_alphabet(label):
    monospaced_characters = ['b','d','g','n','o','p','q','u']
    return monospaced_characters[label]
    # return chr(ord('a') + label)

# Function to write labels to a text file
def write_labels_to_file(labels, filename, width):
    with open(filename, 'w') as file:
        for i, label in enumerate(labels):
            file.write(label_to_alphabet(label))
            if (i + 1) % width == 0:
                file.write('\n')
            # else:
            #     file.write(' ')


# Read the image
image = cv2.imread(r"dwight.jfif")

new_width,new_height=150,150
# resized_image = cv2.resize(image, (new_width, new_height))
image = cv2.resize(image, (new_width, new_height))


(h, w) = image.shape[:2]
print("Height is ",h)
print("Width is ",w)

# Convert the image to LAB color space
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Reshape the image into a feature vector
image_flat = image.reshape((h * w, 3))

# Number of clusters
cluster_n = int(input("Enter the number of clusters you want to create "))

# Apply k-means using MiniBatchKMeans
clt = MiniBatchKMeans(n_clusters=cluster_n)
labels = clt.fit_predict(image_flat)

# Write labels to a text file
output_file = "output_labels1.txt"
write_labels_to_file(labels, output_file, w)

# Print message and display the original and resized images
print(f"Labels written to {output_file}")
cv2.imshow("Original Image", image)
cv2.waitKey(0)
