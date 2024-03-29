import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# dictionary of points
points = {"blue": [[2, 4, 3], [1, 3, 5], [2, 3, 1], [3, 2, 3], [2, 1, 6]],
          "red": [[5, 6, 5], [4, 5, 2], [4, 6, 1], [6, 6, 1], [5, 4, 6], [10, 10, 4]]}

# coordinates of a new point for prediction
new_point = [4, 4, 4]


# euclidian distance for any dimensions
def euclidian_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


class KNN:

    def __init__(self, k=3):
        self.k = k
        self.point = None

    # method to train the KNN model
    def fit(self, points):
        self.points = points

    # method to predict the class of a new point
    def predict(self, new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                distance = euclidian_distance(point, new_point)
                distances.append([distance, category])

        # selection k nearest categories
        categories = [category[1] for category in sorted(distances)[:self.k]]
        # determining the most common category among k nearest neighbors
        result = Counter(categories).most_common(1)[0][0]
        return result


clf = KNN()
clf.fit(points)
print(clf.predict(new_point))

# Visualization

fig = plt.figure(figsize=(15, 21))
ax = fig.add_subplot(projection="3d")
ax.grid(True, color='#323232')
ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

for point in points['blue']:
    ax.scatter(point[0], point[1], point[2], color='#104DCA', s=60)

for point in points['red']:
    ax.scatter(point[0], point[1], point[2], color='#FF0000', s=60)

new_class = clf.predict(new_point)
color = '#FF0000' if new_class == 'red' else '#104DCA'
ax.scatter(new_point[0], new_point[1], point[2], color=color,
           marker='*', s=200, zorder=100)

for point in points['blue']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]],
            color='#104DCA', linestyle='--', linewidth=1)

for point in points['red']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]],
            color='#FF0000', linestyle='--', linewidth=1)

plt.show()
