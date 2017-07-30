# symbol.py
# Ben Zager
# Symbol class for paper2latex

import numpy as np
from tools import hog,lbp,resizeImg


class Symbol:

	def __init__(self,props,img,size=60,extra=4):

		self.props = props # 
		self.label = props.label # 

		self.image = props.image # binary image
		self.box = props.bbox  # (min_row,min_col,max_row,max_col)
		self.coords = props.coords # 
		self.height = props.bbox[2] - props.bbox[0] # 
		self.width = props.bbox[3] - props.bbox[1] # 

		self.original = None # 
		self.square = None # 

		self.setOriginal(img) # 
		self.setSquare(size=size,extra=extra) # 

		# features
		self.centroid = props.centroid # 
		self.area = props.area # 

		self.hog,self.hogImg = None # 
		self.lbp = None # 
	
		self.parent = None # 
		self.children = [] # 


	# Finds segemented region of original grayscale image
	def setOriginal(self,img):
		minr,minc,maxr,maxc = self.box
		self.original = np.ones([maxr-minr,maxc-minc])
		r = self.coords[:,0]
		c = self.coords[:,1]
		self.original[r-minr,c-minc] = img[r,c]

	# Pads and resizes image to square of given size
	def setSquare(self,size=60,extra=4):
		self.square = resizeImg(self.image,size=size,extra=extra)

	# histogram of oriented gradients
	def hog(self,orientations=8,cell=(5,5)):
		self.hog,self.hogImg = hog(self.square,orientations=orientations,cell=cell)

	# local binary pattern
	def lbp(self,P=8,R=1,method="uniform"):
		self.lbp = lbp(self.square,P=P,R=R,method=method)


	# Returns title for plotting
	def getTitle(self):
		label = "Label: "+str(self.label)+" "
		centroid = " ("+str(np.around(self.centroid[0],2))+","+str(np.around(self.centroid[1],2))+") "
		area = "A: "+str(self.area)+" "
		height = "H: "+str(self.height)+" "
		width = "W: "+str(self.width)+" "

		return label+area+height+width+centroid


"""
***Possibly useful attributes***

bbox_area : int Number of pixels of bounding box.
convex_area : int Number of pixels of convex hull image.
eccentricity : float Eccentricity of the ellipse of same second-moments
equivalent_diameter : float Diameter of circle with the same area as the region.
euler_number : int Euler characteristic of region. Computed as number of objects (= 1) subtracted by number of holes (8-connectivity).
extent : float Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows*cols)
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
weighted_moments : (3,3) ndarray Spatial moments of intensity image up to 3rd order:
weighted_moments_central : (3,3) ndarray Central moments (translation invariant) of intensity image up to 3rd order:
weighted_moments_hu : tuple Hu moments (translation, scale and rotation invariant) of intensity image.
weighted_moments_normalized : (3,3) ndarray Normalized moments (translation and scale invariant) of intensity image up to 3rd order:
"""

