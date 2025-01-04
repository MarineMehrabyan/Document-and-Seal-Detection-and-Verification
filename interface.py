from separate import separate  
from extract import extract  
import cv2
import numpy as np
import joblib
from tkinter import Tk, PhotoImage, Label, Button, IntVar, Scale, HORIZONTAL, filedialog, messagebox, LabelFrame, Radiobutton, font, ttk, W
from PIL import Image, ImageTk
from skimage.feature import hog


SIZE = 224

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing App")
        master.geometry("1200x1000")
        master.configure(background="#25292e")
        master.minsize(1400, 1000)  
        master.maxsize(1400, 1000)
        self.font_style = ("Verdana", 12, "bold")
        self.style = ttk.Style()
        self.configure_styles()
        self.image_path = None
        self.image = None
        self.hog_scaler = joblib.load('signature_models/hog_scaler.pkl')
        self.voting_classifier = joblib.load('signature_models/voting_classifier_model.pkl')
        self.best_xgb_model = joblib.load('stamp_models/best_xgb_model.joblib')
        self.create_widgets()

    def update_color_scale(self, value):
        if self.image is not None:
            self.process_image()
           
    def configure_styles(self):
        # Define colors
        base_bg_color = "#25292e"
        base_text_color = "#ededf5"
        accent_color = "#809196"
        border_color = "#42484a"

        self.style.configure('Custom.TRadiobutton',
                             background=base_bg_color,
                             foreground=base_text_color,
                             font=self.font_style,
                             padding=[10, 5],
                             indicatorsize=20, 
                             relief="flat", 
                             bordercolor=border_color,
                             lightcolor=border_color,  
                             darkcolor=border_color,  
                             selectcolor=accent_color)

        self.style.configure('Custom.TButton',
                             foreground="#0c0d0d",
                             background="#d8e4ed",
                             font=self.font_style,
                             relief="flat",
                             padding=[12, 6],
                             focuscolor=base_bg_color, 
                             highlightcolor=base_bg_color, 
                             highlightbackground=accent_color)

        self.style.configure('Custom.TFrame',
                             background=base_bg_color,
                             bordercolor=border_color,
                             relief="groove",
                             padding=10)

    def create_widgets(self):
        title_label = ttk.Label(self.master, text="Document stamp and signature expertise system", font=("Helvetica", 24, "bold"), foreground="#ededf5", background="#25292e")
        title_label.pack(pady=20)
        radio_frame = ttk.Frame(self.master, padding=10, style="Custom.TFrame")
        radio_frame.pack(side="left", padx=20, pady=20, anchor="nw")
        self.radio_var = IntVar()
        options = [("Detect Automatic Extraction", 0), ("Separate Stamps and Signature", 1),
                   ("Check Signature Validation", 2), ("Check Stamp Validation", 3)]
        for index, (text, value) in enumerate(options):
            ttk.Radiobutton(radio_frame, text=text, variable=self.radio_var, value=value, style='Custom.TRadiobutton', command=self.toggle_function(index)).pack(side="top", anchor="w", padx=10, pady=(5, 0))

        control_frame = ttk.Frame(self.master, padding=20, style="Custom.TFrame")
        control_frame.pack(side="top", fill="x", padx=20, pady=(0, 20))
        ttk.Button(control_frame, text="Load Image", command=self.load_image, style='Custom.TButton').pack(side="left", padx=(0, 10))
        self.process_button = ttk.Button(control_frame, text="Process", command=self.process_image, state="disabled", style='Custom.TButton')
        self.process_button.pack(side="left")
        self.color_scale_frame = LabelFrame(self.master, text="Color Scale", padx=10, pady=10, background="#25292e", foreground="#ededf5")
        self.color_scale_frame.pack(side="top", padx=10, pady=10, fill="x")
        self.color_scale = Scale(self.color_scale_frame, from_=0, to=255, orient=HORIZONTAL, label="Color Scale", variable=self.color_scale_frame, command=self.update_color_scale,  length=400)   
        self.color_scale.pack(side="left", padx=(0, 100)) 
        self.color_scale.pack_forget() 
    
        style = ttk.Style()
        style.configure('ImageFrame.TFrame', background="#e9f0f5")
        
        image_frame = ttk.Frame(self.master, padding=20, borderwidth=2, relief="groove",  style="Custom.TFrame")
        image_frame.pack(side="top", fill="both", expand=True, padx=20, pady=20)

        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill="both", expand=True)

    def toggle_function(self, index):
        if index == 1:  # If "Separate Stamps and Signature" is selected
            return lambda: self.toggle(index, show_color_scale=True) 
        else:
            return lambda: self.toggle(index, show_color_scale=False) 


    def toggle(self, index, show_color_scale=False):
        if show_color_scale:
            #self.color_scale_frame.pack(side="top", padx=10, pady=10, fill="x")
            self.color_scale.pack(side="left", padx=(0, 10))  # Initially pack the color scale inside the label frame
            
            self.toggle_separate()
        else:
            self.color_scale.pack_forget()
            self.color_scale_frame.pack_forget() # Hide the color scale

            if index == 0:
                self.toggle_automatic_extraction()
            elif index == 2:
                self.toggle_signature_validation()
            elif index == 3:
                self.toggle_stamp_validation()

    def toggle_automatic_extraction(self):
        self.reset_image_display()
        self.radio_var.set(0)
        self.process_button["state"] = "normal"
      
    def toggle_separate(self):
        self.reset_image_display()
        self.radio_var.set(1)
        self.process_button["state"] = "normal"


    def toggle_signature_validation(self):
        self.reset_image_display()
        self.radio_var.set(2)
        self.process_button["state"] = "normal"

    def toggle_stamp_validation(self):
        self.reset_image_display()
        self.radio_var.set(3)
        self.process_button["state"] = "normal"

    def reset_image_display(self):
        self.image_path = None
        self.image = None
        self.modified_image = None
        self.image_label.config(image="")
        self.process_button["state"] = "disabled"

    def load_image(self):
        options = dict(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        file_path = filedialog.askopenfilename(**options)
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            self.image = cv2.resize(self.image, (self.image_label.winfo_width(), self.image_label.winfo_height()))  # Adjusted image size
            self.modified_image = self.image.copy()
            self.display_image()
            self.process_button["state"] = "normal"

    def display_image(self):
        if self.image is not None:
            q_img = self.convert_image_to_photoimage(self.modified_image)
            self.image_label.config(image=q_img)
            self.image_label.image = q_img 
            
    def process_image(self, event = None):
        if self.radio_var.get() == 0:
            self.detect_automatically()
        elif self.radio_var.get() == 1:
            self.separate_stamps_and_signature()
        elif self.radio_var.get() == 2:
            self.validate_signature()
        elif self.radio_var.get() == 3:
            self.validate_stamp()
        self.display_image()


    def validate_signature(self):
        if self.image_path is not None:
            signature_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            signature_image = cv2.resize(signature_image, (SIZE, SIZE))
            signature_hog_features = hog(signature_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            signature_hog_features_scaled = self.hog_scaler.transform(signature_hog_features.reshape(1, -1))
            prediction = self.voting_classifier.predict(signature_hog_features_scaled)
            if prediction == 0:
                messagebox.showinfo("Signature Validation", "The signature is classified as REAL.")
            else:
                messagebox.showinfo("Signature Validation", "The signature is classified as FORGED.")
        else:
            messagebox.showerror("Error", "Please load an image first.")

    def validate_stamp(self):
        if self.image_path is None:
            messagebox.showerror("Error", "Image path is incorrect.")
        else:
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))  # Resize image to a fixed size (adjust as needed)
            img_flat = img.flatten().reshape(1, -1)
            img_normalized = img_flat / 255.0
            prediction = self.best_xgb_model.predict(img_normalized)
            if prediction[0] == 0:
                messagebox.showinfo("Stamp Validation", "The stamp is classified as REAL.")
            else:
                messagebox.showinfo("Stamp Validation", "The stamp is classified as FAKE.")

    def detect_automatically(self):
        if self.image is not None:
            self.modified_image = extract(self.image) 
            self.display_image()


    def separate_stamps_and_signature(self):

        if self.image is not None:
            original_image = self.image
            scale_value = self.color_scale.get() 
            print(scale_value)
            if  scale_value != 0:
                sign_final_image, stamp_final_image = separate(original_image, scale_value)
            else:
                sign_final_image, stamp_final_image = separate(original_image)
            max_width = self.image_label.winfo_width()
            max_height = self.image_label.winfo_height()
        
            if sign_final_image is not None and not sign_final_image.size == 0:
                sign_final_image = cv2.resize(sign_final_image, (max_width // 3, max_height))
        
            if stamp_final_image is not None and not stamp_final_image.size == 0:  
                stamp_final_image = cv2.resize(stamp_final_image, (max_width // 3, max_height))
        
            combined_image = np.concatenate((sign_final_image, stamp_final_image), axis=1)
            combined_image = cv2.resize(combined_image, (max_width, max_height))
        
            self.modified_image = combined_image
            self.display_image()
        else:
            messagebox.showerror("Error", "Please load an image first.")

            
    def convert_image_to_photoimage(self, image):
        if image is None:
            return PhotoImage()
        height, width = image.shape[:2]
        if len(image.shape) == 2: 
            q_img = Image.fromarray(image) 
            q_img = ImageTk.PhotoImage(q_img) 
        else:  
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            q_img = Image.fromarray(image_rgb) 
            q_img = ImageTk.PhotoImage(q_img)  
 
        return q_img


root = Tk()
app = ImageProcessingApp(root)
root.mainloop()
