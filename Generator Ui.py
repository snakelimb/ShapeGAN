import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageTk
import os

# Import the generator class from your training script
# Make sure this matches your trained model architecture
class DeterministicGenerator(nn.Module):
    """Deterministic Generator that takes only parameters + shape type (no noise)"""
    
    def __init__(self):
        super(DeterministicGenerator, self).__init__()
        
        # Input size: n_params (parameters) + n_shapes (shape type)
        n_params = 8
        n_shapes = 5
        nc = 3
        input_size = n_params + n_shapes
        
        # Projection layer to get to proper size for transposed convolutions
        self.projection = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024 * 4 * 4),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Transposed convolution layers
        self.main = nn.Sequential(
            # Input: 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: 512 x 8 x 8
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 16 x 16
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 32 x 32
            
            nn.ConvTranspose2d(128, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: nc x 64 x 64
        )
    
    def forward(self, parameters, shape_type):
        # Concatenate parameters and shape type (no noise)
        x = torch.cat([parameters, shape_type], dim=1)
        
        # Project to proper size
        x = self.projection(x)
        x = x.view(x.size(0), 1024, 4, 4)
        
        # Generate image
        output = self.main(x)
        return output

class ShapeGeneratorUI:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.shape_names = ["circle", "polygon", "star", "wavy_bar", "cross"]
        
        # Initialize the UI
        self.setup_ui()
        
        # Try to load the model
        self.load_model()
        
        # Generate initial image
        self.update_image()
    
    def setup_ui(self):
        """Create the tkinter interface"""
        self.root = tk.Tk()
        self.root.title("Real-time Shape Generator")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Shape Parameters", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel for image display
        image_frame = ttk.LabelFrame(main_frame, text="Generated Shape (4x Zoom)", padding="10")
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Shape type selection
        ttk.Label(control_frame, text="Shape Type:").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        self.shape_var = tk.StringVar(value="circle")
        for i, shape in enumerate(self.shape_names):
            ttk.Radiobutton(control_frame, text=shape.replace("_", " ").title(), 
                           variable=self.shape_var, value=shape,
                           command=self.update_image).grid(row=i+1, column=0, columnspan=2, sticky=tk.W)
        
        # Parameter sliders
        self.param_vars = []
        slider_configs = [
            ("X Position", 0.0, 1.0, 0.5),
            ("Y Position", 0.0, 1.0, 0.5),
            ("Rotation", 0.0, 1.0, 0.0),
            ("Size", 0.1, 1.0, 0.5),
            ("Red", 0.0, 1.0, 0.5),
            ("Green", 0.0, 1.0, 0.3),
            ("Blue", 0.0, 1.0, 0.8),
            ("Points/Complexity", 0.1, 1.0, 0.5)
        ]
        
        start_row = len(self.shape_names) + 2
        for i, (label, min_val, max_val, default_val) in enumerate(slider_configs):
            # Label
            ttk.Label(control_frame, text=f"{label}:").grid(row=start_row + i*2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
            
            # Slider
            var = tk.DoubleVar(value=default_val)
            self.param_vars.append(var)
            
            slider = ttk.Scale(control_frame, from_=min_val, to=max_val, 
                             variable=var, orient=tk.HORIZONTAL, length=200,
                             command=lambda val, idx=i: self.on_slider_change(idx, val))
            slider.grid(row=start_row + i*2 + 1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
            
            # Value label
            value_label = ttk.Label(control_frame, text=f"{default_val:.3f}")
            value_label.grid(row=start_row + i*2 + 1, column=2, sticky=tk.W, padx=(5, 0))
            
            # Store reference to update the label
            setattr(self, f"value_label_{i}", value_label)
        
        # Configure control frame grid
        control_frame.columnconfigure(0, weight=1)
        
        # Image display
        self.image_label = ttk.Label(image_frame, text="Loading model...")
        self.image_label.grid(row=0, column=0, padx=20, pady=20)
        
        # Configure image frame grid
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def on_slider_change(self, slider_idx, value):
        """Handle slider value changes"""
        # Update the value label
        value_label = getattr(self, f"value_label_{slider_idx}")
        value_label.config(text=f"{float(value):.3f}")
        
        # Update the generated image
        self.update_image()
    
    def load_model(self):
        """Load the trained generator model"""
        model_path = "deterministic_generator_final.pth"
        
        if not os.path.exists(model_path):
            self.status_var.set(f"Error: Model file '{model_path}' not found!")
            messagebox.showerror("Model Not Found", 
                               f"Could not find the trained model at '{model_path}'.\n"
                               "Please make sure you have trained the model first.")
            return False
        
        try:
            self.generator = DeterministicGenerator().to(self.device)
            self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
            self.generator.eval()
            self.status_var.set(f"Model loaded successfully! Using device: {self.device}")
            return True
            
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            messagebox.showerror("Model Loading Error", f"Failed to load model:\n{str(e)}")
            return False
    
    def get_current_parameters(self):
        """Get current parameter values from sliders"""
        return torch.tensor([var.get() for var in self.param_vars], 
                          dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def get_current_shape_type(self):
        """Get current shape type as one-hot encoding"""
        shape_name = self.shape_var.get()
        shape_id = self.shape_names.index(shape_name)
        
        shape_onehot = torch.zeros(5)
        shape_onehot[shape_id] = 1.0
        return shape_onehot.unsqueeze(0).to(self.device)
    
    def update_image(self):
        """Generate and display updated image"""
        if self.generator is None:
            return
        
        try:
            # Get current parameters and shape type
            params = self.get_current_parameters()
            shape_type = self.get_current_shape_type()
            
            # Generate image
            with torch.no_grad():
                generated_tensor = self.generator(params, shape_type)
                
                # Convert to numpy and denormalize
                img_np = generated_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                img_np = (img_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
                img_np = np.clip(img_np, 0, 1)
                
                # Convert to PIL Image
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                
                # Apply 4x zoom using NEAREST neighbor to keep sharp edges
                img_zoomed = img_pil.resize((256, 256), Image.NEAREST)
                
                # Convert to PhotoImage for tkinter
                self.photo = ImageTk.PhotoImage(img_zoomed)
                
                # Update the label
                self.image_label.config(image=self.photo, text="")
                
            # Update status
            shape_name = self.shape_var.get().replace("_", " ").title()
            self.status_var.set(f"Generated {shape_name} - Real-time updates active")
            
        except Exception as e:
            self.status_var.set(f"Error generating image: {str(e)}")
            print(f"Error in update_image: {e}")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    """Main function to run the Shape Generator UI"""
    print("Starting Shape Generator UI...")
    
    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create and run the UI
    app = ShapeGeneratorUI()
    app.run()

if __name__ == "__main__":
    main()