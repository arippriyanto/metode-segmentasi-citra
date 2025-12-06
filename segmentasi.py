import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.cluster import KMeans

# =========================
# Fungsi Segmentasi
# =========================
def segment_diskontinuitas(img_gray):
    # Contoh sederhana: Canny edge detection
    edges = cv2.Canny(img_gray, 100, 200)
    return edges

def segment_thresholding(img_gray):
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

def segment_region_growing(img_gray, seed_point=(0,0)):
    h, w = img_gray.shape
    segmented = np.zeros_like(img_gray)
    to_visit = [seed_point]
    visited = set()
    threshold = 10
    
    while to_visit:
        x, y = to_visit.pop(0)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        segmented[x, y] = 255
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                nx, ny = x+dx, y+dy
                if 0<=nx<h and 0<=ny<w and (nx,ny) not in visited:
                    if abs(int(img_gray[nx,ny])-int(img_gray[x,y]))<threshold:
                        to_visit.append((nx, ny))
    return segmented

def segment_split_merge(img_gray, min_size=32, threshold=10):
    h, w = img_gray.shape
    segmented = np.zeros_like(img_gray)
    
    def split(x, y, size):
        region = img_gray[x:x+size, y:y+size]
        if size <= min_size or region.std() < threshold:
            segmented[x:x+size, y:y+size] = region.mean()
        else:
            half = size//2
            split(x, y, half)
            split(x, y+half, half)
            split(x+half, y, half)
            split(x+half, y+half, half)
    
    split(0, 0, min(h, w))
    return segmented.astype(np.uint8)

def segment_clustering(img_gray, k=2):
    flat = img_gray.flatten().reshape(-1,1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(flat)
    clustered = kmeans.labels_.reshape(img_gray.shape)
    clustered = (clustered * (255//(k-1))).astype(np.uint8)
    return clustered

# =========================
# GUI Application
# =========================
class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation GUI")
        self.root.geometry("900x600")
        
        # Image containers
        self.img = None
        self.img_gray = None
        self.display_img = tk.Label(self.root)
        self.display_img.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Controls
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Load image button
        self.load_btn = tk.Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(pady=5)
        
        # Segmentation method selection
        self.method_label = tk.Label(self.control_frame, text="Segmentation Method")
        self.method_label.pack(pady=5)
        
        self.method_var = tk.StringVar(value="diskontinuitas")
        methods = [("Diskontinuitas","diskontinuitas"), 
                   ("Thresholding","thresholding"), 
                   ("Region Growing","region_growing"),
                   ("Split & Merge","split_merge"),
                   ("Clustering","clustering")]
        
        for text, mode in methods:
            tk.Radiobutton(self.control_frame, text=text, variable=self.method_var, value=mode).pack(anchor=tk.W)
        
        # Apply button
        self.apply_btn = tk.Button(self.control_frame, text="Apply Segmentation", command=self.apply_segmentation)
        self.apply_btn.pack(pady=20)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp")])
        if file_path:
            self.img = cv2.imread(file_path)
            self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.show_image(self.img)
    
    def show_image(self, img_to_show):
        img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB) if len(img_to_show.shape)==3 else cv2.cvtColor(img_to_show, cv2.COLOR_GRAY2RGB)
        im_pil = Image.fromarray(img_rgb)
        im_pil = im_pil.resize((500,500))
        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.display_img.imgtk = imgtk
        self.display_img.configure(image=imgtk)
    
    def apply_segmentation(self):
        if self.img_gray is None:
            messagebox.showerror("Error","Please load an image first!")
            return
        
        method = self.method_var.get()
        if method == "diskontinuitas":
            result = segment_diskontinuitas(self.img_gray)
        elif method == "thresholding":
            result = segment_thresholding(self.img_gray)
        elif method == "region_growing":
            result = segment_region_growing(self.img_gray, seed_point=(self.img_gray.shape[0]//2, self.img_gray.shape[1]//2))
        elif method == "split_merge":
            result = segment_split_merge(self.img_gray)
        elif method == "clustering":
            result = segment_clustering(self.img_gray, k=3)
        else:
            result = self.img_gray
        
        self.show_image(result)

# =========================
# Run the App
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
