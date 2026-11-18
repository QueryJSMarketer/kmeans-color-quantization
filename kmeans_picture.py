import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path

exit_folder = Path(__file__).parent / 'images'
exit_folder.mkdir(exist_ok=True)

r_img = Path(__file__).parent / 'torres-del-paine.png'
img = plt.imread(r_img)
print(img.shape)

img_rgb = img[:, :, :3]
print(img_rgb.shape)

plt.imshow(img_rgb)
plt.axis('off');
plt.show()

img_flat = img_rgb.reshape(img_rgb.shape[0] * img_rgb.shape[1], 3)
print(img_flat[:5, :])
print()
print(img_flat.shape)

img_flat_df = pd.DataFrame(img_flat, columns=['r', 'g', 'b'])
print(img_flat_df.head())

sample_df = img_flat_df.sample(n=5000, random_state=42)
print(sample_df.head(5))

rgb_colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b in zip(sample_df['r'], sample_df['g'], sample_df['b'])]


fig = go.Figure(data=[go.Scatter3d(x=sample_df['r'],
                                   y=sample_df['g'],
                                   z=sample_df['b'],
                                   mode='markers',
                                   marker = dict(size=3, color=rgb_colors, opacity=0.8))])

fig.update_layout(margin=dict(l=0, 
                              r=0, 
                              b=0, 
                              t=0),
                  scene=dict(
    xaxis_title='R',
    yaxis_title='G',
    zaxis_title='B',
))

fig.write_image(exit_folder / 'rgb_space_3d.png', width=900, height=600, scale=2)
fig.show()

"""
In the 3D graph, 
using a sample of 5,000 pixels, 
we can see the color "fingerprint" of your photograph distributed within an RGB cube, 
revealing which tones predominate and how they are mathematically grouped.
• The three axes (R, G, B): These represent the intensity of Red, Green, and Blue for each pixel, 
ranging from 0.0 (the complete absence of that color) to 1.0 (maximum intensity).
• Corner (0, 0, 0): This is where the deep shadows and black tones of a photo are concentrated (such as dark mountain rocks).
• Corner (1, 1, 1): This is where the brightest, whitest areas appear (such as snow on mountain peaks or brilliant clouds).
• Central diagonal: The imaginary line connecting black (0, 0, 0) to white (1, 1, 1) contains all the gray and neutral tones.
"""

"""
Clustered point clouds (natural clusters): Instead of seeing points scattered randomly throughout the cube, 
we will notice well-defined patches or clusters (for example, a bluish/light-blue cloud for the sky and the lake, 
a grayish/brownish one for the rocky massif, and a white one for the snow).
"""

"""
K means:
• Visual justification for K-Means: This chart demonstrates why K-Means works. 
Since pixels representing similar elements fall very close to one another in 3D space, 
the algorithm can easily find the "center" of each cluster of points, 
thereby summarizing the entire photograph into a few key colors.
"""
kmeans = KMeans(n_clusters= 1, random_state= 30).fit(img_flat)
img_flat1 = img_flat.copy()

for i in np.unique(kmeans.labels_):
    img_flat1[kmeans.labels_==i,:] = kmeans.cluster_centers_[i]

img1 = img_flat1.reshape(img_rgb.shape)

print(img1[:5, :])
plt.imshow(img1)
plt.axis('off');
plt.imsave(exit_folder / 'kmeans_k1.png', img1)
plt.show()

"""
Next, we update the RGB values ​​of each pixel so that they match those of the centroid. 
The result is an image of Torres del Paine in which each pixel has been replaced by the average color. 
The average color of this photograph turned out to be gray—a blend of all the colors. 
We can verify this ourselves by manually calculating the average of each column in the flattened array. 
This will provide us with the average R, G, and B values.
"""
#calculate the mean of each column in the flattened array
column_mean = img_flat.mean(axis=0)
print('Column means: ', column_mean)

#Now, we can compare this with what the K-means model calculated as the final location of its single centroid.
print('Clusters mean: ', kmeans.cluster_centers_)

#They are not strictly identical in terms of decimal places, but rather a very good approximation.

centroid = kmeans.cluster_centers_[0].tolist()

fig.add_trace(go.Scatter3d(x=[centroid[0]],
y=[centroid[1]],
z=[centroid[2]],
mode='markers',
marker=dict(size=7, color=[f'rgb({int(centroid[0]*255)}, {int(centroid[1]*255)}, {int(centroid[2]*255)})'], opacity=1)))

fig.show()
"""
We can observe the centroid as a large circle in the middle of the color space. 
(If you cannot see it, simply click on the image and rotate or zoom in on it.) 
Keep in mind that this represents the "center of gravity" of all the points in the graph.

Now, let’s try something different. 
Let’s fit a K-means model to the data again, this time using k = 3.
"""
print()
kmeans_3 = KMeans(n_clusters= 3, random_state= 30).fit(img_flat)
img_flat_3 = img_flat.copy()

for i in np.unique(kmeans_3.labels_):
    img_flat_3[kmeans_3.labels_==i,:] = kmeans_3.cluster_centers_[i]

img_3 = img_flat_3.reshape(img_rgb.shape)
print(img_3[:5, :])
plt.imshow(img_3)
plt.axis('off');
plt.imsave(exit_folder / 'kmeans_k3.png', img_3)
plt.show()

centers = kmeans_3.cluster_centers_
print(centers)
print((centers * 255).astype(int))
"""
Now we will create a helper function to easily display RGB values ​​as color swatches, 
and we will use it to check the colors of the model's centroids.
"""

def show_watch(RGB_value):
    R, G, B = RGB_value
    rgb = [[np.array([R * 255, G *255, B * 255]).astype('uint8')]]
    plt.figure()
    plt.imshow(rgb)
    plt.axis('off')
    plt.show();

for pixel in centers:
    show_watch(pixel)

"""
Al igual que antes, 
vamos a reemplazar cada píxel de la imagen original por el valor RGB del centroide al que fue asignado.
"""
def cluster_image(k, img=img_rgb):
    img_flat = img.reshape(img.shape[0] * img.shape[1], 3)
    kmeans = KMeans(n_clusters = k, random_state = 42).fit(img_flat)
    new_img = img_flat.copy()

    for i in np.unique(kmeans.labels_):
        new_img[kmeans.labels_==i, :] = kmeans.cluster_centers_[i]
    
    new_img = new_img.reshape(img.shape)
    return plt.imshow(new_img), plt.axis('off'), plt.show();

cluster_image(3);

"""
Now we have an image composed of only three colors. 
The RGB values ​​of each pixel correspond to the values ​​of its nearest centroid.

We can return, once again, to our three-dimensional color space. 
This time, we will change the color of each point in that space to match the color of its centroid. 
This will allow us to observe how the K-means algorithm grouped our data spatially.
"""

print(kmeans_3.labels_.shape)
print(kmeans_3.labels_)
print(np.unique(kmeans_3.labels_))
print(kmeans_3.cluster_centers_)

img_flat_df['cluster'] = kmeans_3.labels_
img_flat_df.head()

series_conversion = {i: f'rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})'
                     for i, c in enumerate(kmeans_3.cluster_centers_)}

print(series_conversion)

"""
Replace the cluster numbers in the 'cluster' column with formatted RGB values
"""
img_flat_df['cluster'] = img_flat_df['cluster'].map(series_conversion)
sample_flat_df = img_flat_df.sample(n=5000, random_state=40)
img_flat_df.head()

"""
Re-plot the data, now showing which cluster (i.e., color) each point was assigned to by K-means when k=3.
"""
trace = go.Scatter3d(x = sample_flat_df.r,
                     y = sample_flat_df.g,
                     z = sample_flat_df.b,
                     mode='markers',
                     marker=dict(size=3,
                                 color=sample_flat_df.cluster,
                                 opacity=0.8))

layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=0))

fig = go.Figure(data=[trace], layout=layout)
fig.write_image(exit_folder / 'rgb_clusters_3d.png', width=900, height=600, scale=2)
fig.show()
plt.show()

def cluster_image_grid(k, ax, img=img_rgb):
    img_flat = img.reshape(img.shape[0] * img.shape[1], 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(img_flat)
    new_img = img_flat.copy()
    
    for i in np.unique(kmeans.labels_):
        new_img[kmeans.labels_==i, :] = kmeans.cluster_centers_[i]
    
    new_img = new_img.reshape(img.shape)
    ax.imshow(new_img)
    ax.axis('off')
    
fig, axs = plt.subplots(3, 3, figsize=(9, 12))
axs = axs.flatten()
k_values = np.arange(2, 11)
for i, k in enumerate(k_values):
    cluster_image_grid(k, axs[i], img=img_rgb)
    axs[i].title.set_text('k=' + str(k))

plt.tight_layout()
plt.savefig(exit_folder / 'kmeans_grid_k2_k10.png', dpi=300, bbox_inches='tight')
plt.show()


