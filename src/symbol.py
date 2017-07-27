# symbol.py
# Ben Zager
# Symbol class for paper2latex

import numpy as np

class Symbol:

	def __init__(self,props):

		self.props = props

		self.image = props.image
		self.label = props.label
		self.centroid = props.centroid
		self.area = props.area

		self.parent = None
		self.children = []

	def getTitle(self):
		label = "Label: "+str(self.label)
		centroid = " ("+str(self.centroid[0])+","+str(self.centroid[1])+") "

		return label+centroid

"""
other props attributes

bbox : tuple Bounding box (min_row, min_col, max_row, max_col). Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col).
bbox_area : int Number of pixels of bounding box.
convex_area : int Number of pixels of convex hull image.
convex_image : (H,J) ndarray Binary convex hull image which has the same size as bounding box.
coords : (N,2) ndarray Coordinate list (row, col) of the region.
eccentricity : float Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
equivalent_diameter : float The diameter of a circle with the same area as the region.
euler_number : int Euler characteristic of region. Computed as number of objects (= 1) subtracted by number of holes (8-connectivity).
extent : float Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
filled_area : int Number of pixels of filled region.
filled_image : (H,J) ndarray Binary region image with filled holes which has the same size as bounding box.
inertia_tensor : (2,2) ndarray Inertia tensor of the region for the rotation around its mass.
inertia_tensor_eigvals : tuple The two eigen values of the inertia tensor in decreasing order.
local_centroid : array Centroid coordinate tuple (row,col), relative to region bounding box.
major_axis_length : float The length of the major axis of the ellipse that has the same normalized second central moments as the region.
minor_axis_length : float The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
moments : (3,3) ndarray Spatial moments up to 3rd order:
moments_central : (3,3) ndarray Central moments (translation invariant) up to 3rd order:
moments_hu : tuple Hu moments (translation, scale and rotation invariant).
moments_normalized : (3,3) ndarray Normalized moments (translation and scale invariant) up to 3rd order:
orientation : float Angle between the X-axis and the major axis of the ellipse that has the same second-moments as the region. Ranging from -pi/2 to pi/2 in counter-clockwise direction.
perimeter : float Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
solidity : float Ratio of pixels in the region to pixels of the convex hull image.
weighted_centroid : array Centroid coordinate tuple (row, col) weighted with intensity image.
weighted_local_centroid : array Centroid coordinate tuple (row, col), relative to region bounding box, weighted with intensity image.
weighted_moments : (3, 3) ndarray Spatial moments of intensity image up to 3rd order:
weighted_moments_central : (3, 3) ndarray Central moments (translation invariant) of intensity image up to 3rd order:
weighted_moments_hu : tuple Hu moments (translation, scale and rotation invariant) of intensity image.
weighted_moments_normalized : (3, 3) ndarray
Normalized moments (translation and scale invariant) of intensity image up to 3rd order:
"""

