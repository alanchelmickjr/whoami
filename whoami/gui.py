"""
GUI Module for Face Recognition System
Simple and portable GUI using tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
from typing import Optional
from .face_recognizer import FaceRecognizer


class FaceRecognitionGUI:
    """Main GUI application for face recognition"""
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI
        
        Args:
            root: tkinter root window
        """
        self.root = root
        self.root.title("WhoAmI - Facial Recognition System")
        self.root.geometry("1000x700")
        
        # Initialize face recognizer
        self.recognizer = FaceRecognizer()
        
        # State variables
        self.running = False
        self.camera_active = False
        self.current_frame = None
        self.add_mode = False
        self.pending_name = None
        
        # Create UI
        self.create_widgets()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Video feed
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Right panel - Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera controls
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Controls", padding="10")
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(
            camera_frame, text="Start Camera", command=self.start_camera
        )
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(
            camera_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED
        )
        self.stop_button.pack(fill=tk.X, pady=2)
        
        # Face management
        face_frame = ttk.LabelFrame(control_frame, text="Face Management", padding="10")
        face_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.add_face_button = ttk.Button(
            face_frame, text="Add Face", command=self.add_face_dialog, state=tk.DISABLED
        )
        self.add_face_button.pack(fill=tk.X, pady=2)
        
        self.remove_face_button = ttk.Button(
            face_frame, text="Remove Face", command=self.remove_face_dialog
        )
        self.remove_face_button.pack(fill=tk.X, pady=2)
        
        self.clear_button = ttk.Button(
            face_frame, text="Clear All Faces", command=self.clear_all_faces
        )
        self.clear_button.pack(fill=tk.X, pady=2)
        
        # Known faces list
        list_frame = ttk.LabelFrame(control_frame, text="Known Faces", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.faces_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.faces_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.faces_listbox.yview)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Update faces list
        self.update_faces_list()
    
    def start_camera(self):
        """Start the camera and video feed"""
        self.status_var.set("Starting camera...")
        
        if self.recognizer.start_camera():
            self.camera_active = True
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.add_face_button.config(state=tk.NORMAL)
            
            # Start video update thread
            self.video_thread = threading.Thread(target=self.update_video, daemon=True)
            self.video_thread.start()
            
            self.status_var.set("Camera active")
        else:
            messagebox.showerror("Error", "Failed to start camera. Make sure Oak D is connected.")
            self.status_var.set("Camera start failed")
    
    def stop_camera(self):
        """Stop the camera and video feed"""
        self.running = False
        self.camera_active = False
        
        if self.recognizer:
            self.recognizer.stop_camera()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.add_face_button.config(state=tk.DISABLED)
        self.status_var.set("Camera stopped")
    
    def update_video(self):
        """Update video feed in a separate thread"""
        while self.running:
            frame = self.recognizer.get_frame()
            
            if frame is not None:
                self.current_frame = frame.copy()
                
                # Detect and recognize faces
                face_locations, face_encodings = self.recognizer.detect_faces(frame)
                
                if not self.add_mode:
                    # Recognition mode
                    recognized_faces = self.recognizer.recognize_faces(face_encodings)
                    
                    # Draw rectangles and labels
                    for (top, right, bottom, left), (name, confidence) in zip(
                        face_locations, recognized_faces
                    ):
                        # Draw rectangle
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Draw label
                        label = f"{name}"
                        if name != "Unknown":
                            label += f" ({confidence:.2f})"
                        
                        cv2.rectangle(
                            frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED
                        )
                        cv2.putText(
                            frame, label, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1
                        )
                else:
                    # Add mode - draw rectangles around detected faces
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                        cv2.putText(
                            frame, "Position face here", (left, top - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1
                        )
                
                # Convert frame to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize to fit display
                img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image=img)
                
                # Update label
                self.video_label.configure(image=photo)
                self.video_label.image = photo
    
    def add_face_dialog(self):
        """Show dialog to add a new face"""
        name = simpledialog.askstring("Add Face", "Enter person's name:")
        
        if name:
            self.pending_name = name
            self.add_mode = True
            self.status_var.set(f"Position face in frame and click 'Capture'")
            
            # Show capture dialog
            self.show_capture_dialog()
    
    def show_capture_dialog(self):
        """Show capture dialog window"""
        capture_window = tk.Toplevel(self.root)
        capture_window.title("Capture Face")
        capture_window.geometry("300x100")
        
        label = ttk.Label(
            capture_window, 
            text=f"Position {self.pending_name}'s face in the camera view",
            wraplength=280
        )
        label.pack(pady=10)
        
        button_frame = ttk.Frame(capture_window)
        button_frame.pack(pady=10)
        
        def capture():
            if self.current_frame is not None:
                if self.recognizer.add_face(self.pending_name, self.current_frame):
                    messagebox.showinfo("Success", f"Face added for {self.pending_name}")
                    self.update_faces_list()
                    self.status_var.set(f"Added face: {self.pending_name}")
                else:
                    messagebox.showerror(
                        "Error", 
                        "Could not detect face. Ensure one face is clearly visible."
                    )
                    self.status_var.set("Face capture failed")
            
            self.add_mode = False
            self.pending_name = None
            capture_window.destroy()
        
        def cancel():
            self.add_mode = False
            self.pending_name = None
            self.status_var.set("Capture cancelled")
            capture_window.destroy()
        
        ttk.Button(button_frame, text="Capture", command=capture).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
    
    def remove_face_dialog(self):
        """Show dialog to remove a face"""
        names = self.recognizer.get_all_names()
        
        if not names:
            messagebox.showinfo("Info", "No faces in database")
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Remove Face")
        dialog.geometry("300x200")
        
        ttk.Label(dialog, text="Select face to remove:").pack(pady=10)
        
        listbox = tk.Listbox(dialog)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for name in sorted(names):
            listbox.insert(tk.END, name)
        
        def remove():
            selection = listbox.curselection()
            if selection:
                name = listbox.get(selection[0])
                if messagebox.askyesno("Confirm", f"Remove {name} from database?"):
                    if self.recognizer.remove_face(name):
                        messagebox.showinfo("Success", f"Removed {name}")
                        self.update_faces_list()
                        self.status_var.set(f"Removed face: {name}")
                        dialog.destroy()
                    else:
                        messagebox.showerror("Error", "Failed to remove face")
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Remove", command=remove).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def clear_all_faces(self):
        """Clear all faces from database"""
        if messagebox.askyesno("Confirm", "Remove all faces from database?"):
            self.recognizer.clear_database()
            self.update_faces_list()
            self.status_var.set("All faces cleared")
            messagebox.showinfo("Success", "All faces removed from database")
    
    def update_faces_list(self):
        """Update the list of known faces"""
        self.faces_listbox.delete(0, tk.END)
        names = self.recognizer.get_all_names()
        for name in sorted(names):
            self.faces_listbox.insert(tk.END, name)
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.camera_active:
            self.recognizer.stop_camera()
        self.root.destroy()


def main():
    """Main entry point for GUI application"""
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
