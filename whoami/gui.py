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
        
        # Face selection for adding
        self.detected_face_locations = []
        self.selected_face_index = None
        self.selected_face_location = None
        
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
        
        # Bind click event for face selection
        self.video_label.bind("<Button-1>", self.on_video_click)
        
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
                
                # Store detected face locations for selection
                self.detected_face_locations = face_locations
                
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
                    # Add mode - draw rectangles around detected faces with selection highlight
                    for idx, (top, right, bottom, left) in enumerate(face_locations):
                        # Check if this face is selected
                        if self.selected_face_index == idx:
                            # Selected face - thick green border
                            color = (0, 255, 0)
                            thickness = 4
                            label = "SELECTED - Click Capture"
                        else:
                            # Unselected face - yellow border
                            color = (255, 255, 0)
                            thickness = 2
                            label = "Click to select"
                        
                        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
                        
                        # Draw label background
                        cv2.rectangle(
                            frame, (left, top - 25), (right, top), color, cv2.FILLED
                        )
                        cv2.putText(
                            frame, label, (left + 6, top - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1
                        )
                
                # Convert frame to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Store original size for click mapping
                self.video_original_size = img.size
                
                # Resize to fit display
                img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                self.video_display_size = img.size
                
                photo = ImageTk.PhotoImage(image=img)
                
                # Update label on main thread using after()
                self.root.after(0, self._update_video_label, photo)
    
    def _update_video_label(self, photo):
        """Update video label on main thread (thread-safe)"""
        self.video_label.configure(image=photo)
        self.video_label.image = photo
    
    def on_video_click(self, event):
        """Handle click on video to select a face"""
        if not self.add_mode or not self.detected_face_locations:
            return
        
        # Get click coordinates relative to the label
        click_x, click_y = event.x, event.y
        
        # Scale coordinates to match original frame size
        if hasattr(self, 'video_display_size') and hasattr(self, 'video_original_size'):
            scale_x = self.video_original_size[0] / self.video_display_size[0]
            scale_y = self.video_original_size[1] / self.video_display_size[1]
            
            # Convert to original frame coordinates
            orig_x = click_x * scale_x
            orig_y = click_y * scale_y
            
            # Find which face was clicked
            for idx, (top, right, bottom, left) in enumerate(self.detected_face_locations):
                if left <= orig_x <= right and top <= orig_y <= bottom:
                    self.selected_face_index = idx
                    self.selected_face_location = (top, right, bottom, left)
                    self.status_var.set(f"Selected face {idx + 1} of {len(self.detected_face_locations)}")
                    break
    
    def add_face_dialog(self):
        """Show dialog to add a new face"""
        name = simpledialog.askstring("Add Face", "Enter person's name (leave empty for auto-numbering):")
        
        if name is not None:  # User clicked OK (even if empty string)
            self.pending_name = name if name.strip() else None
            self.add_mode = True
            self.selected_face_index = None
            self.selected_face_location = None
            
            if self.pending_name:
                self.status_var.set(f"Click on {self.pending_name}'s face, then click 'Capture'")
            else:
                self.status_var.set("Click on a face to select it, then click 'Capture' (auto-numbering)")
            
            # Show capture dialog
            self.show_capture_dialog()
    
    def show_capture_dialog(self):
        """Show capture dialog window"""
        capture_window = tk.Toplevel(self.root)
        capture_window.title("Capture Face")
        capture_window.geometry("400x150")
        
        if self.pending_name:
            instruction_text = f"Click on {self.pending_name}'s face in the camera view,\nthen click 'Capture'"
        else:
            instruction_text = "Click on a face in the camera view to select it,\nthen click 'Capture' for auto-numbering"
        
        label = ttk.Label(
            capture_window,
            text=instruction_text,
            wraplength=380
        )
        label.pack(pady=10)
        
        # Status label for face selection
        self.capture_status_label = ttk.Label(
            capture_window,
            text="No face selected",
            foreground="red"
        )
        self.capture_status_label.pack(pady=5)
        
        # Update capture status periodically
        def update_capture_status():
            if self.selected_face_index is not None:
                self.capture_status_label.config(
                    text=f"Face {self.selected_face_index + 1} selected",
                    foreground="green"
                )
            else:
                self.capture_status_label.config(
                    text="No face selected - click on a face",
                    foreground="red"
                )
            
            if capture_window.winfo_exists():
                capture_window.after(100, update_capture_status)
        
        update_capture_status()
        
        button_frame = ttk.Frame(capture_window)
        button_frame.pack(pady=10)
        
        def capture():
            if self.current_frame is not None:
                if self.selected_face_index is None:
                    messagebox.showwarning(
                        "No Selection",
                        "Please click on a face to select it first."
                    )
                    return
                
                # Add only the selected face
                success = False
                name_to_add = self.pending_name
                
                # Get all face locations and encodings
                face_locations, face_encodings = self.recognizer.detect_faces(self.current_frame)
                
                if self.selected_face_index < len(face_encodings):
                    # Auto-generate name if not provided
                    if not name_to_add:
                        # Count existing unknown faces to generate the next number
                        existing_names = self.recognizer.get_all_names()
                        unknown_count = sum(1 for name in existing_names if name.startswith("unknown_"))
                        name_to_add = f"unknown_{unknown_count + 1}"
                    
                    # Add the selected face encoding directly
                    self.recognizer.known_face_encodings.append(face_encodings[self.selected_face_index])
                    self.recognizer.known_face_names.append(name_to_add)
                    self.recognizer.save_database()
                    success = True
                    
                    if success:
                        messagebox.showinfo("Success", f"Face added as '{name_to_add}'")
                        self.update_faces_list()
                        self.status_var.set(f"Added face: {name_to_add}")
                else:
                    messagebox.showerror(
                        "Error",
                        "Selected face is no longer detected. Please try again."
                    )
                    self.status_var.set("Face capture failed")
            
            self.add_mode = False
            self.pending_name = None
            self.selected_face_index = None
            self.selected_face_location = None
            capture_window.destroy()
        
        def cancel():
            self.add_mode = False
            self.pending_name = None
            self.selected_face_index = None
            self.selected_face_location = None
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
        
        # If only one face, remove it directly with confirmation
        if len(names) == 1:
            name = names[0]
            if messagebox.askyesno("Remove Face", f"Remove {name} from database?"):
                if self.recognizer.remove_face(name):
                    self.update_faces_list()
                    self.status_var.set(f"Removed face: {name}")
                else:
                    messagebox.showerror("Error", "Failed to remove face")
            return
        
        # Multiple faces - show selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Remove Face")
        dialog.geometry("300x200")
        
        # No need for redundant "Select face to remove" label - title says it all
        
        listbox = tk.Listbox(dialog)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for name in sorted(names):
            listbox.insert(tk.END, name)
        
        # Select first item by default
        listbox.selection_set(0)
        
        def remove():
            selection = listbox.curselection()
            if selection:
                name = listbox.get(selection[0])
                # Remove the redundant confirmation - user already selected and clicked Remove
                if self.recognizer.remove_face(name):
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
