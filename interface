
import tkinter as tk
import dicom2nifti.convert_dicom as convert_dicom
from PIL import Image, ImageTk
import os
import pydicom
import sys
from tkinter import ttk, filedialog, simpledialog, Menu
import nibabel as nib
from tkinter import Menu
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tkinter import Label
import re
from PIL import Image, ImageFilter
from front import *
import dicom2nifti
import tempfile
import shutil 
sequences={}
current_folder_path = ""
# Liste des chemins d'accès aux images
image_paths = []
# Image sélectionnée
selected_image = None
# Index de l'image actuelle dans la liste des chemins d'accès aux images
current_image_index = 0
# Création de la fenêtre princ
def display_patient_information(file_path):
    # Effacer le contenu précédent du tableau
    table.delete(*table.get_children())

    # Lire les données DICOM à partir du fichier image
    ds = pydicom.dcmread(file_path)

    patient_name = ds.PatientName
    patient_id = ds.PatientID
    patient_sex = ds.PatientSex
    patient_age = ds.PatientAge

    # Extraire les informations du patient
    study_date = ds.StudyDate
    study_time = ds.StudyTime
    study_id = ds.StudyID
    study_modality = ds.Modality
    study_description = ds.StudyDescription
    series_date = ds.SeriesDate
    series_time = ds.SeriesTime
    series_description = ds.SeriesDescription

    table.insert("", "end", values=("Patient Name", patient_name))
    table.insert("", "end", values=("Patient ID", patient_id))
    table.insert("", "end", values=("Patient Sex", patient_sex))
    table.insert("", "end", values=("Patient Age", patient_age))

    table.insert("", "end", values=("Study Date", study_date))
    table.insert("", "end", values=("Study Time", study_time))
    table.insert("", "end", values=("Study ID", study_id))
    table.insert("", "end", values=("Study Modality", study_modality))
    table.insert("", "end", values=("Study Description", study_description))
    table.insert("", "end", values=("Series Date", series_date))
    table.insert("", "end", values=("Series Time", series_time))
    table.insert("", "end", values=("Series Description", series_description))

def extract_dicom_tags(file_path):
    ds = pydicom.dcmread(file_path)

    table.delete(*table.get_children())

    tags = []

    for attr_name in ds:
        tag_key = str(attr_name.tag)
        tag_description = str(attr_name.name)
        tag_value = str(attr_name.value)
        tags.append((tag_key, tag_description, tag_value))

    for tag_key, tag_description, tag_value in tags:
        table.insert("", "end", values=(tag_key, tag_description, tag_value))



root = tk.Tk()
root.title("Interface Tkinter ")
current_folder_path = ""

left_frame = ttk.Frame(root)
middle_frame = ttk.Frame(root)
right_frame = ttk.Frame(root)

# Grid layout for the frames
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
middle_frame.grid(row=0, column=1, padx=10, pady=10)
right_frame.grid(row=0, column=2, padx=10, pady=10, sticky="ns")


image_paths = []
def browse_folder():
    global current_folder_path, sequences
    folder_path = filedialog.askdirectory()
    if folder_path:
        current_folder_path = folder_path
        sequences = classifier_sequences(current_folder_path)
        display_sequences()
        

def exit_program():
    root.quit()
    root.destroy()
    sys.exit()
def apply_gaussian_blur(label):
    global selected_image

    if selected_image is not None:
        # Convertir l'image sélectionnée en tableau NumPy avec le bon type de données
        image_array = np.array(selected_image, dtype=np.uint8)
        blurred_image = cv2.GaussianBlur(image_array, (11, 11), 0)
        blurred_pil_image = Image.fromarray(blurred_image)  # Créez l'image PIL à partir de l'array flou
        label.image = ImageTk.PhotoImage(blurred_pil_image)
        label.configure(image=label.image)
        label.pack()
def apply_average_blur(image_label):
    global selected_image

    if selected_image is not None:
        
        image_array = np.array(selected_image, dtype=np.uint8)
        blurred_image = cv2.blur(image_array, (5, 5))
        blurred_pil_image = Image.fromarray(blurred_image)
        image_label.image = ImageTk.PhotoImage(blurred_pil_image)
        image_label.configure(image=image_label.image)
        image_label.pack()
def apply_laplacian(image_label):
    global selected_image

    if selected_image is not None:
        # Convertir l'image PIL en tableau NumPy
        image_array = np.array(selected_image)

        # Appliquer le filtre Laplacian
        laplacian_image = cv2.Laplacian(image_array, cv2.CV_64F)

        # Convertir l'image filtrée en image PIL
        laplacian_pil_image = Image.fromarray(laplacian_image.astype(np.uint8))

        # Créer un nouvel objet ImageTk.PhotoImage pour l'image filtrée
        laplacian_image_tk = ImageTk.PhotoImage(laplacian_pil_image)

        # Mettre à jour l'image dans image_label
        image_label.configure(image=laplacian_image_tk)
        image_label.image = laplacian_image_tk
        image_label.pack()
def apply_motion_blur(image_label):
    global selected_image

    if selected_image is not None:
        # Convert the selected image to a NumPy array with the correct data type
        image_array = np.array(selected_image, dtype=np.uint8)
        kernel_size = 7
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Apply the motion blur filter
        blurred_image = cv2.filter2D(image_array, -1, kernel)
        blurred_pil_image = Image.fromarray(blurred_image)  # Convert the blurred array back to a PIL image

        image_label.image = ImageTk.PhotoImage(blurred_pil_image)
        image_label.configure(image=image_label.image)
        image_label.pack()


def segment_current_image():
    global selected_image

    if selected_image is not None:
        # Convertir l'image sélectionnée en tableau NumPy
        image_array = np.array(selected_image)

        # Effectuer la segmentation avec GMM
        masked_img, contour_image = perform_segmentation(image_array)

        # Convertir l'image segmentée en image PIL
        segmented_pil_image = Image.fromarray(contour_image)

        # Mettre à jour l'image dans image_label_middle
        image_label.image = ImageTk.PhotoImage(segmented_pil_image)
        image_label.configure(image=image_label.image)
        image_label.pack()
def wissem():
    global selected_image

    if selected_image is not None:
        # Convertir l'image sélectionnée en tableau NumPy
        image_array = np.array(selected_image)

        # Effectuer la segmentation avec GMM
        masked_image, contour_image, filled = perform_geodesic_active_contour_segmentation(image_array)

        # Convertir l'image segmentée en image PIL
        segmented_pil_image = Image.fromarray(filled)

        # Mettre à jour l'image dans image_label_middle
        image_label.image = ImageTk.PhotoImage(segmented_pil_image)
        image_label.configure(image=image_label.image)
        image_label.pack()
def update_image_label(image):
    global image_label

    image_array = np.array(image)

    final_edges = canny_edge_detection(image_array)

    # Convert the final_edges array back to an image
    final_edges_image = Image.fromarray((final_edges * 255).astype(np.uint8))

    image_label.image = ImageTk.PhotoImage(final_edges_image)
    image_label.configure(image=image_label.image)
    image_label.pack()

def brain_mask():
    global selected_image

    if selected_image is not None:
        # Convertir l'image sélectionnée en tableau NumPy
        image_array = np.array(selected_image)

        # Effectuer la segmentation avec GMM
        masked_img, contour_image = perform_segmentation(image_array)

        # Convertir l'image segmentée en image PIL
        segmented_pil_image = Image.fromarray(masked_img)

        # Mettre à jour l'image dans image_label_middle
        image_label.image = ImageTk.PhotoImage(segmented_pil_image)
        image_label.configure(image=image_label.image)
        image_label.pack()
def segment_gmm():
    global selected_image

    if selected_image is not None:
        # Convertir l'image sélectionnée en tableau NumPy
        image_array = np.array(selected_image)

        # Effectuer la segmentation avec GMM
        segmented = gmm(image_array)

        # Convertir les labels de cluster en une image en niveaux de gris
        segmented_gray = (segmented * 255 / segmented.max()).astype(np.uint8)

        # Convertir l'image segmentée en image PIL
        segmented_pil_image = Image.fromarray(segmented_gray, mode='L')

        image_label.image = ImageTk.PhotoImage(segmented_pil_image)
        image_label.configure(image=image_label.image)
        image_label.pack()
def segment_with_kmeans():
    global selected_image

    if selected_image is not None:
        # Convertir l'image PIL en tableau NumPy
        image_array = np.array(selected_image)
        num_clusters = 3  # Vous pouvez ajuster le nombre de clusters
        segmented = segment_kmeans(image_array, num_clusters)

        # Convertir l'image segmentée en image PIL
        segmented_pil_image = Image.fromarray((segmented * 255 / segmented.max()).astype(np.uint8))

        # Afficher l'image segmentée dans votre interface utilisateur Tkinter
        image_label.image = ImageTk.PhotoImage(segmented_pil_image)
        image_label.configure(image=image_label.image)
        image_label.pack()


def get_image_paths():
    image_paths = []

    # Parcourir l'arborescence pour obtenir les chemins d'accès aux images
    for patient_node in tree.get_children():
        for seq_type_node in tree.get_children(patient_node):
            for image_node in tree.get_children(seq_type_node):
                file_name = tree.item(image_node)['text']
                file_path = os.path.join(current_folder_path, file_name)
                image_paths.append(file_path)

    return image_paths

def classifier_sequences(folder_path):
    sequences = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.dcm'):
            file_path = os.path.join(folder_path, filename)
            dcm = pydicom.dcmread(file_path)

            # Vérification de l'existence de la séquence
            if hasattr(dcm, 'SequenceName') and hasattr(dcm, 'SeriesDescription'):
                sequence_name = dcm.SequenceName
                sequence_name1 = dcm.SeriesDescription
                if '*ep_b0' == sequence_name:
                    seq_type = 'b0'
                elif '*ep_b1000t' == sequence_name:
                    seq_type = 'b100'
                elif 'ep2d_diff_b0_b1000_trace_p2_ADC' == sequence_name1:
                    seq_type = 'ADC'
                else:
                    continue  # Séquence non reconnue, passer à la prochaine itération

                # Vérification de l'existence du nom du patient
                if hasattr(dcm, 'PatientName'):
                    patient_name = dcm.PatientName
                    if patient_name not in sequences:
                        sequences[patient_name] = {}

                    # Ajouter le chemin d'accès au fichier DICOM à la séquence correspondante
                    if seq_type not in sequences[patient_name]:
                        sequences[patient_name][seq_type] = []

                    sequences[patient_name][seq_type].append(file_path)

    # Afficher les chemins d'accès pour chaque séquence de chaque patient
    for patient_name, seq_info in sequences.items():
        print(f"Patient: {patient_name}")
        for seq_type, file_paths in seq_info.items():
            print(f"  Sequence {seq_type}:")
            for file_path in file_paths:
                print(f"    {file_path}")
    
    return sequences


def display_sequences():
    global image_paths, sequences  # Ajoutez 'sequences' ici
    tree.delete(*tree.get_children())
    image_paths = get_image_paths()

    for patient, seq_types in sequences.items():
        patient_node = tree.insert('', 'end', text=patient, open=True)

        for seq_type, seq_files in seq_types.items():
            seq_type_node = tree.insert(patient_node, 'end', text=seq_type, open=True)

            for file_path in seq_files:
                tree.insert(seq_type_node, 'end', text=os.path.basename(file_path))

def show_previous_image():
    global current_image_index, selected_image, image_paths

    image_paths = get_image_paths()

    if current_image_index > 0:
        current_image_index -= 1
        print("Previous image index:", current_image_index)  # Debugging line
        file_path = image_paths[current_image_index]
        selected_image = load_dicom_image(file_path)

        image_label.image = ImageTk.PhotoImage(selected_image)
        image_label.configure(image=image_label.image)
        image_label.config(text=file_path)

        # Mettre à jour le chemin sélectionné dans l'arborescence
        select_treeview_item(file_path)

def show_next_image():
    global current_image_index, selected_image, image_paths

    image_paths = get_image_paths()

    if current_image_index < len(image_paths) - 1:
        current_image_index += 1
        print("Next image index:", current_image_index)  # Debugging line
        file_path = image_paths[current_image_index]
        selected_image = load_dicom_image(file_path)

        image_label.image = ImageTk.PhotoImage(selected_image)
        image_label.configure(image=image_label.image)
        image_label.config(text=file_path)

        # Mettre à jour le chemin sélectionné dans l'arborescence
        select_treeview_item(file_path)


def select_treeview_item(file_path):
    # Parcourir les éléments de l'arborescence pour trouver le chemin correspondant
    for patient_node in tree.get_children():
        for seq_type_node in tree.get_children(patient_node):
            for image_node in tree.get_children(seq_type_node):
                item_file_path = os.path.join(current_folder_path, tree.item(image_node)['text'])
                if item_file_path == file_path:
                    # Sélectionner le chemin dans l'arborescence
                    tree.selection_set(image_node)
                    # Défiler jusqu'à l'élément sélectionné
                    tree.see(image_node)
                    return
def save_image_as_jpg():
    img = img_label.image
    file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
    
    if file_path:
        img.save(file_path, "JPEG")
        print("L'image a été enregistrée en tant que", file_path)


# Fonction pour obtenir la liste des séquences disponibles dans le répertoire actuel
def choose_sequence():
    seq_types = ["b0", "b100", "ADC"]

    # Loop until the user enters a valid sequence type
    while True:
        seq_type = simpledialog.askstring("Select Sequence Type", "Choose a sequence type:", initialvalue="b0")

        if seq_type in seq_types:
            return seq_type
        else:
            print("Invalid sequence type. Please enter 'b0', 'b100', or 'ADC'.")

# Fonction pour afficher une boîte de dialogue de sélection d'emplacement
def choose_save_location():
    return filedialog.asksaveasfilename(defaultextension=".nii.gz", filetypes=[("NIfTI files", "*.nii.gz")])
# Fonction pour convertir et sauvegarder une séquence au format NIfTI
def convert_and_save_sequence(patient_name, seq_type):
    global sequences

    # Vérifier si le patient existe dans le dictionnaire
    if patient_name in sequences:
        patient_info = sequences[patient_name]

        # Vérifier si la séquence existe pour ce patient
        if seq_type in patient_info:
            dicom_file_paths = patient_info[seq_type]

            if not dicom_file_paths:
                print("No DICOM images found for the selected sequence type.")
                return

            # Créer un répertoire temporaire pour les fichiers DICOM
            temp_dir = tempfile.mkdtemp()

            try:
                # Copier les fichiers DICOM dans le répertoire temporaire
                for dicom_file_path in dicom_file_paths:
                    shutil.copy(dicom_file_path, os.path.join(temp_dir, os.path.basename(dicom_file_path)))

                # Convertir les fichiers DICOM en fichiers NIfTI dans le répertoire temporaire
                nifti_file_path = os.path.join(temp_dir, patient_name + "_" + seq_type + ".nii.gz")
                convert_dicom.dicom_series_to_nifti(temp_dir, nifti_file_path)

                # Demander à l'utilisateur de spécifier l'emplacement de sauvegarde pour le fichier NIfTI
                output_folder = filedialog.askdirectory(title="Choose Output Folder")
                if not output_folder:
                    print("No output folder selected.")
                    return

                # Copier le fichier NIfTI du répertoire temporaire vers l'emplacement choisi par l'utilisateur
                shutil.copy(nifti_file_path, output_folder)

                print("NIfTI image saved in:", output_folder)

            finally:
                # Supprimer le répertoire temporaire et son contenu
                shutil.rmtree(temp_dir)
        else:
            print(f"No sequence of type {seq_type} found for patient {patient_name}.")
    else:
        print(f"No information found for patient {patient_name}.")
def download_nifti():
    global sequences
    # Vérifier s'il y a des informations de séquence disponibles
    if not sequences:
        print("No sequence information available.")
        return

    # Demander à l'utilisateur de choisir le nom du patient
    patient_name = simpledialog.askstring("Select Patient", "Choose a patient name:")

    if not patient_name:
        print("No patient selected.")
        return

    # Vérifier si le patient existe dans les informations de séquence
    if patient_name in sequences:
        # Demander à l'utilisateur de choisir le type de séquence
        seq_type = simpledialog.askstring("Select Sequence Type", "Choose a sequence type :b0, b100, ADC:", initialvalue="b0")

        if not seq_type:
            print("No sequence type selected.")
            return

        # Appeler la fonction pour convertir et sauvegarder la séquence sélectionnée
        convert_and_save_sequence(patient_name, seq_type)
    else:
        print(f"No information found for patient {patient_name}.")



menubar = Menu(root, tearoff=False)
root.config(menu=menubar)  # Associer le menu à la fenêtre
menu_file = Menu(menubar)
menu_edit = Menu(menubar)
menu_view = Menu(menubar)
menu_Filter = Menu(menubar)
menu_aide = Menu(menubar)
menu_seg = Menu(menubar)
menu_help = Menu(menubar)
menubar.add_cascade(menu=menu_file, label='File')
menu_file.add_command(label="Add Directory", command=browse_folder)

menu_file.add_separator()
menu_file.add_command(label="Exit", command=exit_program)

menubar.add_cascade(menu=menu_view, label='View')
menubar.add_cascade(menu=menu_Filter, label='Filter')
menubar.add_cascade(menu=menu_seg, label='Segmentation')
menubar.add_cascade(menu=menu_aide, label='Diagnostic assistance')
menubar.add_cascade(menu=menu_help, label='Help')

menu_blur = Menu(menu_Filter)
menu_Filter.add_cascade(label="Blur", menu=menu_blur)
menu_Filter.add_separator()
menu_blur.add_command(label="Gaussian", command=lambda: apply_gaussian_blur(image_label))
menu_blur.add_command(label="Motion", command=lambda: apply_motion_blur(image_label))
menu_blur.add_command(label="Average", command=lambda: apply_average_blur(image_label))
menu_view.add_command(label="Télécharger en PNG", command=lambda: save_current_image(image_label))
menu_view.add_command(label="Télécharger NIfTI",command=download_nifti)
menu_seg.add_command(label="Chan-vese",command=wissem)
menu_seg.add_command(label="GMM",command=segment_gmm)
menu_seg.add_command(label="Kmeans",command=segment_with_kmeans)
# Création du sous-menu "Edge Sharpness" avec les types de filtres
menu_edge = Menu(menu_Filter)
menu_Filter.add_command(label="Canny",  command=lambda: update_image_label(selected_image))
menu_Filter.add_command(label="Laplacian",  command=lambda: apply_laplacian(image_label))
menu_Filter.add_separator()
#menu_edge.add_command(label="Sobel")#, command=apply_sobel_filter)
#menu_edge.add_command(label="Canny",  command=lambda: update_image_label(selected_image))
#menu_edge.add_command(label="Laplacian")#, command=apply_laplacian_filter)
menu_Filter.add_cascade(label="Brain mask", command=brain_mask)
# Partie gauche : TreeView pour les séquences d'IRM

# Configure grid weights for left_frame
left_frame.columnconfigure(0, weight=1)
left_frame.rowconfigure(0, weight=0)  # Set weight to 0 for the title label row
left_frame.rowconfigure(1, weight=1)  # Set weight to 1 for the Treeview row

# Create a title label for the left_frame
tree_label = ttk.Label(left_frame, text="IRM Sequence", font=("Helvetica", 16, "bold"))
tree_label.grid(row=0, column=0, columnspan=3, pady=10)

# Create a Treeview for the left_frame
tree = ttk.Treeview(left_frame)
tree.grid(row=1, column=0, sticky="nsew")
# Create vertical scrollbar and link it to the Treeview
scroll_vertical = ttk.Scrollbar(left_frame, orient="vertical", command=tree.yview)
scroll_vertical.grid(row=1, column=1, sticky="ns")
tree.configure(yscrollcommand=scroll_vertical.set)

# Create horizontal scrollbar and link it to the Treeview
scroll_horizontal = ttk.Scrollbar(left_frame, orient="horizontal", command=tree.xview)
scroll_horizontal.grid(row=2, column=0, sticky="ew")
tree.configure(xscrollcommand=scroll_horizontal.set)

# Créer les boutons Précédent et Suivant
previous_button = tk.Button(left_frame, text="previous", command=show_previous_image)
next_button = tk.Button(left_frame, text="next", command=show_next_image)

# Positionnement des boutons
previous_button.grid(row=3, column=0, padx=(0,5), pady=(10, 3), sticky='e')
next_button.grid(row=3, column=1, padx=(0,0), pady=(10, 3), sticky='e')

# Chargez l'image par défaut (vous devrez la remplacer par une image réelle)
default_image = Image.open("C:\\Users\\Lenovo\\Downloads\\cerveau.png")

new_width = 512  
new_height = 512 
default_image = default_image.resize((new_width, new_height), Image.LANCZOS)
default_image = ImageTk.PhotoImage(default_image)

image_label = ttk.Label(middle_frame, image=default_image)
image_label.pack()
last_selected_file = ""

def on_treeview_select(event):
    global current_folder_path, selected_image, current_image_index, last_selected_file

    # Récupérer l'élément sélectionné dans le Treeview
    selected_item = tree.selection()
    if not selected_item:
        return

    # Récupérer le chemin du fichier à partir de l'élément sélectionné
    file_name = tree.item(selected_item[0])['text']
    file_path = os.path.join(current_folder_path, file_name)

    # Vérifier si le fichier sélectionné est différent du dernier fichier sélectionné
    if file_path != last_selected_file:
        # Mettre à jour le dernier fichier sélectionné
        last_selected_file = file_path

        # Charger l'image sélectionnée
        selected_image = load_dicom_image(file_path)

        # Mettre à jour l'image dans image_label_middle
        image_label.image = ImageTk.PhotoImage(selected_image)
        image_label.configure(image=image_label.image)
        image_label.pack()

        # Mettre à jour la liste des chemins d'accès aux images et l'index de l'image actuelle
        image_paths = get_image_paths()
        current_image_index = image_paths.index(file_path)

        # Mettre à jour les commandes des boutons pour passer le chemin du fichier
        patient_info_button.configure(command=lambda path=file_path: display_patient_information(path))
        all_tags_button.configure(command=lambda path=file_path: extract_dicom_tags(path))

        # Afficher le chemin du fichier sélectionné
        print("Selected File Path:", file_path)


tree.bind("<<TreeviewSelect>>", on_treeview_select)

info_text = tk.StringVar()

# Créer une étiquette avec l'option textvariable liée à la variable info_text
info_label = tk.Label(middle_frame, textvariable=info_text)
info_label.pack()

def on_mouse_move(event):
    if selected_image is None:
        return
    # Obtenir les coordonnées de la souris
    x = event.x
    y = event.y
    
    if x < selected_image.width and y < selected_image.height:
        pixel_value = selected_image.getpixel((x, y))
        info_text.set("Coordonnées : ({}, {}),       Valeur de Pixel : {}".format(x, y, pixel_value))
    

# Associer la fonction on_mouse_move à l'événement de mouvement de la souris sur l'image
image_label.bind("<Motion>", on_mouse_move)
# Partie droite : Table contenant les informations du patient
# Configure grid weights to make the Treeview expand
right_frame.columnconfigure(0, weight=1)
right_frame.rowconfigure(0, weight=0)  # Set weight to 0 for the title label row
right_frame.rowconfigure(1, weight=1)  # Set weight to 1 for the Treeview row

# Create a title label
title_label = ttk.Label(right_frame, text="Dicom Tag", font=("Helvetica", 16, "bold"))
title_label.grid(row=0, column=0, columnspan=3, pady=10)

# Create the Treeview widget
table = ttk.Treeview(right_frame, columns=("Tag", "Description", "Value"), show="headings")
table.grid(row=1, column=0, sticky="nsew")

# Set the width of columns to minimize horizontal size
table.column("Tag", width=80)
table.column("Description", width=110)
table.column("Value", width=100)

# Create vertical scrollbar and link it to the Treeview
scroll_vertical = ttk.Scrollbar(right_frame, orient="vertical", command=table.yview)
scroll_vertical.grid(row=1, column=1, sticky="ns")
table.configure(yscrollcommand=scroll_vertical.set)

# Create horizontal scrollbar and link it to the Treeview
scroll_horizontal = ttk.Scrollbar(right_frame, orient="horizontal", command=table.xview)
scroll_horizontal.grid(row=2, column=0, sticky="ew")
table.configure(xscrollcommand=scroll_horizontal.set)

# Add column headings
table.heading("Tag", text="Tag")
table.heading("Description", text="Description")
table.heading("Value", text="Value")
# Create a button to show tags
patient_info_button = tk.Button(right_frame, text="Patient Information", command=lambda: display_patient_information(file_path))
all_tags_button = tk.Button(right_frame, text="All Tags", command=lambda: extract_dicom_tags(file_path))

# Positionner les boutons
patient_info_button.grid(row=3, column=0, padx=5, pady=5, sticky="e")
all_tags_button.grid(row=3, column=1, padx=5, pady=5, sticky="w")

root.mainloop()
