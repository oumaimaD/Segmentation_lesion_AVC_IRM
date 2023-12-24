import os
import pydicom
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import Menu
from PIL import Image, ImageTk, ImageGrab
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tkinter import Label
import re
from PIL import Image, ImageFilter
import math 
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided




current_folder_path = ""

def load_dicom_image(file_path):
    ds = pydicom.dcmread(file_path)
    image = ds.pixel_array.astype(float)

    # Gestion de l'intercept et de la mise à l'échelle
    if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
        intercept = ds.RescaleIntercept
        slope = ds.RescaleSlope
        if slope != 1:
            image = slope * image
        image = image + intercept

    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

    # Conversion en format compatible avec Tkinter
    image = Image.fromarray((image * 255).astype(np.uint8))

    return image


def perform_geodesic_active_contour_segmentation(input_image):
    def mat_math(input_matrix, operation):
        output = input_matrix.copy()
        for i in range(input_matrix.shape[0]):
            for j in range(input_matrix.shape[1]):
                if operation == "atan":
                    output[i, j] = math.atan(input_matrix[i, j])
                elif operation == "sqrt":
                    output[i, j] = math.sqrt(input_matrix[i, j])
        return output

    def CV(LSF, img, mu, nu, epsilon, step):
        Drc = (epsilon / math.pi) / (epsilon * epsilon + LSF * LSF)
        Hea = 0.6 * (1 + (2 / math.pi) * mat_math(LSF / epsilon, "atan"))
        Iy, Ix = np.gradient(LSF)
        s = mat_math(Ix * Ix + Iy * Iy, "sqrt")
        Nx = Ix / (s + 0.000001)
        Ny = Iy / (s + 0.000001)
        Mxx, Nxx = np.gradient(Nx)
        Nyy, Myy = np.gradient(Ny)
        curvature = Nxx + Nyy
        Length = nu * Drc * curvature

        Lap = cv2.Laplacian(LSF, -1)
        Penalty = mu * (Lap - curvature)

        s1 = Hea * img
        s2 = (1 - Hea) * img
        s3 = 1 - Hea
        C1 = s1.sum() / Hea.sum()
        C2 = s2.sum() / s3.sum()
        CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))

        LSF = LSF + step * (Length + Penalty + CVterm)
        return LSF

    mu = 0.2  # Poids du terme de longueur dans la fonction de Chan-Vese
    nu = 0.1 * 255 * 255  # Poids du terme de surface dans la fonction de Chan-Vese
    num = 200
    epsilon = 1.7
    step = 0.1

    img_gaussian = cv2.GaussianBlur(input_image, (5, 5), 0)
    img_gray_uint8 = cv2.convertScaleAbs(img_gaussian)

    # Créer une image binaire avec OTSU
    ret, thresh = cv2.threshold(img_gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Opérations morphologiques pour nettoyer l'image binaire
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = clear_border(opening)

    # Créer une image "sure_bg"
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    # Calculer la carte de distance
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    # Trouver les contours dans sure_fg
    contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(sure_fg)
    for i in range(len(contours)):
        cv2.drawContours(mask, contours, i, 255, -1)

    # Effectuer l'inpainting
    filled = cv2.inpaint(img_gray_uint8, mask, 3, cv2.INTER_LINEAR)

    IniLSF = np.ones((filled.shape[0], filled.shape[1]), filled.dtype)
    IniLSF[180:350, 180:350] = -1  # 50/160
    IniLSF = -IniLSF
    
    LSF = IniLSF.copy()

    for i in range(1, num):
        LSF = CV(LSF, filled, mu, nu, epsilon, step)
        if i % 1 == 0:
            plt.imshow(filled, cmap='gray')
            plt.xticks([]), plt.yticks([])
            plt.contour(LSF, [0], colors='r', linewidth=2)
            #plt.contour(IniLSF, [0], colors='b', linewidth=2)  # Ajout de cette ligne pour afficher IniLSF en bleu
            plt.draw()
            plt.show(block=False)
            plt.pause(0.01)

    # Déplacez l'affichage final en dehors de la boucle
    plt.imshow(filled, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.contour(LSF, [0], colors='r', linewidth=2)
    #plt.contour(IniLSF, [0], colors='b', linewidth=2)  # Ajout de cette ligne pour afficher IniLSF en bleu
    plt.show()

    masked_image = cv2.bitwise_and(img_gaussian, img_gaussian, mask=mask)
    contour_image = filled.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)

    return masked_image, contour_image, filled
def perform_segmentation(input_img):
	if isinstance(input_img, str):
		input_img = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
	else:
		input_img = np.array(input_img)
	#ds = pydicom.dcmread(dicom_file_path)
	#image_array = ds.pixel_array
	img_gaussian = cv2.GaussianBlur(input_img, (5, 5), 0)
	#img_gray = cv2.cvtColor(img_gaussian, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.convertScaleAbs(img_gaussian)


	ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	kernel = np.ones((3, 3), np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
	opening = clear_border(opening)
	sure_bg = cv2.dilate(opening, kernel, iterations=2)

	dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
	ret, sure_fg = cv2.threshold(dist_transform, 0.17 * dist_transform.max(), 255, 0)
	sure_fg = np.uint8(sure_fg)

	contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	mask = np.zeros_like(sure_fg)
	for i in range(len(contours)):
		cv2.drawContours(mask, contours, i, 255, -1)

	filled = cv2.inpaint(img_gray, mask, 3, cv2.INTER_LINEAR)
	masked_img = cv2.bitwise_and(input_img, input_img, mask=mask)
	image = masked_img
	
	img = np.array(image, dtype=np.float64)
	IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
	IniLSF[180:350, 180:350] = -1 # 50/160
	#IniLSF[190:330, 190:330] = -1 # 50/160
	IniLSF = -IniLSF

	def mat_math(intput, str):
		output = intput
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if str == "atan":
					output[i, j] = math.atan(intput[i, j])
				if str == "sqrt":
					output[i, j] = math.sqrt(intput[i, j])
		return output
	
	def CV(LSF, img, mu, nu, epison, step):
		Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
		Hea = 0.6 * (1 + (2 / math.pi) * mat_math(LSF / epison, "atan"))
		#Hea = 0.5 * (1 + (2 / math.pi) * mat_math(LSF / epison, "atan"))
		Iy, Ix = np.gradient(LSF)
		s = mat_math(Ix * Ix + Iy * Iy, "sqrt")
		Nx = Ix / (s + 0.000001)
		Ny = Iy / (s + 0.000001)
		Mxx, Nxx = np.gradient(Nx)
		Nyy, Myy = np.gradient(Ny)
		cur = Nxx + Nyy
		Length = nu * Drc * cur
	
		Lap = cv2.Laplacian(LSF, -1)
		Penalty = mu * (Lap - cur)
	
		s1 = Hea * img
		s2 = (1 - Hea) * img
		s3 = 1 - Hea
		C1 = s1.sum() / Hea.sum()
		C2 = s2.sum() / s3.sum()
		CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))
	
		LSF = LSF + step * (Length + Penalty + CVterm)
		return LSF
	
	mu = 0.2 # the weight parameter of the length term in the Chan-Vese functional
	nu = 0.08 * 255 * 255 # the weight parameter of the area term in the Chan-Vese functional 0.005
	num = 200
	epison = 1.7 
	step = 0.1
	LSF = IniLSF

	for i in range(1, 200):
		LSF = CV(LSF, img, mu, nu, epison, step)

	contour_image = input_img.copy()
	cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)

	return masked_img, contour_image

def gmm(image):
    
    img_gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    img2 = img_gaussian.reshape((-1, 1))
    gmm_model = GaussianMixture(n_components=3, covariance_type='tied').fit(img2)

    gmm_labels = gmm_model.predict(img2)
    original_shape = img_gaussian.shape
    segmented = gmm_labels.reshape(original_shape[0], original_shape[1])

    return segmented



def segment_kmeans(image, num_clusters=3):
    # Convertir l'image en un tableau 2D de valeurs de pixel
    image_array = image.reshape((-1, 1))

    # Appliquer K-Means pour la segmentation
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(image_array)
    cluster_labels = kmeans.predict(image_array)

    # Remettre les labels dans la forme d'origine
    segmented = cluster_labels.reshape(image.shape)

    return segmented

def save_current_image(image_label):
    # Obtenez l'image actuelle affichée dans le label
    current_image = image_label.image

    if current_image and isinstance(current_image, ImageTk.PhotoImage):
        # Obtenez la source d'origine de l'image
        image_source = current_image._PhotoImage__photo

        # Ouvrez une boîte de dialogue pour sélectionner l'emplacement de sauvegarde
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])

        if file_path:
            # Enregistrez l'image au format PNG
            image_source.write(file_path, format="png")
            print("L'image a été enregistrée avec succès : ", file_path)

# # Appliquer le filtre de Canny à une image
# def apply_canny(image):
   
#     # Appliquer le filtre de Canny pour détecter les contours
#     edges = cv2.Canny(image, threshold1=100, threshold2=200)  # Ajustez les seuils selon vos besoins
    
#     return edges

# Appliquer le filtre Laplacian à une image
# def apply_laplacian(image):
#     # Convertir l'image en niveaux de gris (nécessaire pour Laplacian)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Appliquer le filtre Laplacian pour détecter les contours
#     laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    
#     # Convertir les valeurs négatives en valeurs positives
#     laplacian = cv2.convertScaleAbs(laplacian)
    
#     return laplacian

# # Appliquer les filtres de Sobel à une image
# def apply_sobel(image):
#     # Convertir l'image en niveaux de gris (nécessaire pour Sobel)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Calculer les gradients horizontaux et verticaux avec les filtres de Sobel
#     gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
#     gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
#     # Calculer le gradient total (magnitude) et l'angle
#     gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
#     gradient_angle = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)
    
#     return gradient_magnitude, gradient_angle


def canny_edge_detection(im, w=11, sigma=3.6, hi_ratio=2.5, lo_ratio=0.05):#sigma=3.6,hi=1.6
    # Gaussian kernel generation
    def gaussian_kernel2d(w, sigma):
        w = w + (w % 2 == 0)
        F = np.zeros([w,w])
        mid = w//2 
        k = np.arange(w) - mid
        denom = 2*np.pi*sigma**2
        for i in k:
            for j in k:
                par = (i**2 + j**2)/(2*sigma**2)
                F[i + mid,j + mid] = np.exp(-par)/denom
        return F
    
    # Define the convolution function
    def fastest_convolution_2d(im, K):
        w, _ = K.shape
        m,n = im.shape
        im_all_subw = get_all_window(im, w)
        X = np.sum(np.sum(im_all_subw * K, 1), 1)
        return X.reshape(m,n)
    
    # Function to get all windows from an image
    def get_all_window(M, w):
        M = np.pad(M, w//2, 'symmetric')
        sub_shape = (w, w)
        view_shape = tuple(np.subtract(M.shape, sub_shape) + 1) + sub_shape
        arr_view = as_strided(M, view_shape, M.strides * 2)
        arr_view = arr_view.reshape((-1,) + sub_shape)
        return arr_view
    
    # Function to perform non-maximum suppression
    def non_maximum_suppression(im, theta):
        ntheta = (np.round(theta / 45) * 45) % 180
        thetafilters = np.array([
            [[0,0,0],[-1,2,-1],[0,0,0]],
            [[-1,0,0],[0,2,0],[0,0,-1]],
            [[0,-1,0],[0,2,0],[0,-1,0]],
            [[0,0,-1],[0,2,0],[-1,0,0]]])
        
        per_angle_res = [fastest_convolution_2d(im, thetafilters[i]) * (ntheta==(45*i)) for i in range(4)]
        return np.sum(per_angle_res,0)
    
    # Double thresholding
    def double_thresholding(im, hi, lo):
        strong = im > hi
        weak = (im >= hi) == (im <= lo)
        return strong, weak
    
    # Hysteresis
    def hysteresis_fast(strong, weak):
        union = (strong + weak) > 0
        K = np.array([
                [1,1,1],
                [1,-4,1],
                [1,1,1]
            ])
        return np.bitwise_and(fastest_convolution_2d(union,K)>=0,union)
    
    imgrey = im.astype(int)
    
    # Apply Gaussian filter
    gaussk = gaussian_kernel2d(w, sigma)
    im_gauss = fastest_convolution_2d(imgrey, gaussk)
    
    # Calculate gradient magnitude and direction
    Gx = fastest_convolution_2d(im_gauss, np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
    Gy = fastest_convolution_2d(im_gauss, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    
    # Perform non-maximum suppression
    im_nms = non_maximum_suppression(G, theta)
    
    maxcap = np.max(im_nms)
    hi_tres = hi_ratio * maxcap
    lo_tres = lo_ratio * maxcap
    
    strong_dt, weak_dt = double_thresholding(im_nms, hi_tres, lo_tres)
    
    final_edges = hysteresis_fast(strong_dt, weak_dt)
    
    return final_edges
