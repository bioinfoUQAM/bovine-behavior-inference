"""
02_annotation_correction_tool.py
================================
Thesis section 4.2.3 (verification and manual correction of the annotations).

Tkinter GUI that shows the video frame by frame next to the frame-level
annotations produced by 01_observer_events_to_binary.py, so that aberrant
temporal overlaps between mutually exclusive behaviours and implausible
transitions can be fixed by hand.

Features
--------
  * load a video (mp4/avi/mov/mkv) and an annotation CSV (Frame + one 0/1
    column per behaviour);
  * browse frame by frame or jump to a frame;
  * toggle behaviours with checkboxes or with auto-assigned keyboard
    shortcuts (shown as "Behaviour [K]");
  * apply the checked behaviours to a whole range of frames;
  * save (in place) or "save as".

This step is interactive and optional for reproducing the thesis results:
the corrections are already contained in the Observer export
(..._Events_fixed.xlsx), from which 01_observer_events_to_binary.py rebuilds
exactly the labels used in data/05_trajectories.

Input / output
--------------
    data/01_annotations_binary/<sequence>_behaviors.csv   (edited in place)
    data/videos/<sequence>.mp4

Usage
-----
    python scripts/02_annotation_correction_tool.py
    python scripts/02_annotation_correction_tool.py \
        --video data/videos/<sequence>.mp4 \
        --csv   data/01_annotations_binary/<sequence>_behaviors.csv
"""

import argparse
import cv2
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class AnnotationCorrectionTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotation Correction Tool")
        self.root.geometry("1300x900")
        
        self.video_path = None
        self.csv_path = None
        self.annotations_df = None
        self.cap = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.behavior_columns = []
        self.modified = False
        
        # Keyboard shortcut -> behaviour mapping
        self.keyboard_shortcuts = {}
        
        self.setup_ui()
        
        # Keyboard bindings
        self.root.bind('<Key>', self.handle_keypress)
    
    def setup_ui(self):
        # Main menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load video", command=self.load_video)
        file_menu.add_command(label="Load annotation CSV", command=self.load_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_annotations)
        file_menu.add_command(label="Save as...", command=self.save_annotations_as)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit_app)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Resizing behaviour
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Video panel
        video_frame = ttk.LabelFrame(main_frame, text="Video", padding="10")
        video_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(video_frame, text="No video loaded", relief=tk.SUNKEN)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.frame_info_label = ttk.Label(info_frame, text="Frame: 0 / 0", font=("Arial", 12, "bold"))
        self.frame_info_label.pack()
        
        # Current annotations panel
        current_annotation_frame = ttk.LabelFrame(main_frame, text="Current annotations", padding="10")
        current_annotation_frame.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.current_annotation_text = tk.Text(current_annotation_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.current_annotation_text.pack(fill=tk.BOTH, expand=True)
        
        # Correction panel
        correction_frame = ttk.LabelFrame(main_frame, text="Annotation correction (use the keyboard shortcuts to toggle)", padding="10")
        correction_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Scrollable container for the checkboxes
        canvas = tk.Canvas(correction_frame, height=150)
        scrollbar = ttk.Scrollbar(correction_frame, orient="vertical", command=canvas.yview)
        self.checkbox_frame = ttk.Frame(canvas)
        
        self.checkbox_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.checkbox_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.checkboxes = {}
        
        # Apply to a frame range
        range_frame = ttk.LabelFrame(main_frame, text="Apply to a frame range", padding="10")
        range_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        range_control_frame = ttk.Frame(range_frame)
        range_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(range_control_frame, text="From frame:").pack(side=tk.LEFT, padx=5)
        self.range_start_entry = ttk.Entry(range_control_frame, width=10)
        self.range_start_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(range_control_frame, text="To frame:").pack(side=tk.LEFT, padx=5)
        self.range_end_entry = ttk.Entry(range_control_frame, width=10)
        self.range_end_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(range_control_frame, text="✓ Apply to selected frames", 
                  command=self.apply_to_range, style="Accent.TButton").pack(side=tk.LEFT, padx=10)
        
        ttk.Label(range_control_frame, text="(applies the checked behaviours to every frame of the range)").pack(side=tk.LEFT, padx=5)
        
        # Navigation buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        
        ttk.Button(control_frame, text="◄◄ Previous frame", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="✓ Correct (next frame)", command=self.next_frame, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="✓ OK - Confirm correction", command=self.confirm_correction).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next frame ►►", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        
        # Jump to frame
        jump_frame = ttk.Frame(main_frame)
        jump_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        
        ttk.Label(jump_frame, text="Go to frame:").pack(side=tk.LEFT, padx=5)
        self.jump_entry = ttk.Entry(jump_frame, width=10)
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(jump_frame, text="Go", command=self.jump_to_frame).pack(side=tk.LEFT, padx=5)
        
        # Keyboard shortcuts display
        self.shortcuts_label = ttk.Label(jump_frame, text="", foreground="blue")
        self.shortcuts_label.pack(side=tk.LEFT, padx=20)
    
    def assign_keyboard_shortcuts(self):
        """Automatically assign a keyboard key to every behaviour."""
        self.keyboard_shortcuts = {}
        used_keys = set()
        
        for behavior in self.behavior_columns:
            # Try the first letter (lower case)
            first_letter = behavior[0].lower()
            
            if first_letter not in used_keys and first_letter.isalpha():
                self.keyboard_shortcuts[first_letter] = behavior
                used_keys.add(first_letter)
            else:
                # Otherwise try the other letters of the word
                for char in behavior.lower():
                    if char.isalpha() and char not in used_keys:
                        self.keyboard_shortcuts[char] = behavior
                        used_keys.add(char)
                        break
        
        # Show the shortcuts
        shortcuts_text = "Shortcuts: " + ", ".join([f"{key.upper()}={behavior[:15]}" for key, behavior in sorted(self.keyboard_shortcuts.items())])
        self.shortcuts_label.config(text=shortcuts_text[:100] + "...")
    
    def handle_keypress(self, event):
        """Toggle a behaviour with its keyboard shortcut."""
        key = event.char.lower()
        
        if key in self.keyboard_shortcuts:
            behavior = self.keyboard_shortcuts[key]
            # Toggle the checkbox
            current_value = self.checkboxes[behavior].get()
            self.checkboxes[behavior].set(1 - current_value)
    
    def load_video(self, filepath=None):
        filepath = filepath or filedialog.askopenfilename(
            title="Select the video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filepath:
            self.video_path = filepath
            self.cap = cv2.VideoCapture(filepath)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            messagebox.showinfo("Success", f"Video loaded: {os.path.basename(filepath)}\nTotal frames: {self.total_frames}")
            if self.annotations_df is not None:
                self.display_frame(0)
    
    def load_csv(self, filepath=None):
        filepath = filepath or filedialog.askopenfilename(
            title="Select the annotation CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.csv_path = filepath
            self.annotations_df = pd.read_csv(filepath)
            
            # Behaviour columns = every column except 'Frame'
            self.behavior_columns = [col for col in self.annotations_df.columns if col != 'Frame']
            
            # Build the checkboxes
            self.create_checkboxes()
            
            # Assign keyboard shortcuts
            self.assign_keyboard_shortcuts()
            
            messagebox.showinfo("Success", f"Annotations loaded: {os.path.basename(filepath)}\nFrames: {len(self.annotations_df)}\nBehaviours: {len(self.behavior_columns)}")
            
            if self.cap is not None:
                self.display_frame(0)
    
    def create_checkboxes(self):
        # Remove the previous checkboxes
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()
        
        self.checkboxes = {}
        
        # Checkbox grid (4 columns)
        for idx, behavior in enumerate(self.behavior_columns):
            row = idx // 4
            col = idx % 4
            
            var = tk.IntVar()
            # The shortcut letter is added later
            cb = ttk.Checkbutton(self.checkbox_frame, text=behavior, variable=var)
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
            
            self.checkboxes[behavior] = var
    
    def update_checkbox_labels(self):
        """Add the keyboard shortcut to every checkbox label."""
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()
        
        for idx, behavior in enumerate(self.behavior_columns):
            row = idx // 4
            col = idx % 4
            
            # Shortcut of this behaviour
            shortcut = ""
            for key, beh in self.keyboard_shortcuts.items():
                if beh == behavior:
                    shortcut = f" [{key.upper()}]"
                    break
            
            cb = ttk.Checkbutton(self.checkbox_frame, 
                               text=f"{behavior}{shortcut}", 
                               variable=self.checkboxes[behavior])
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
    
    def display_frame(self, frame_idx):
        if self.cap is None or self.annotations_df is None:
            messagebox.showwarning("Warning", "Please load the video and the annotations first.")
            return
        
        if frame_idx < 0 or frame_idx >= len(self.annotations_df):
            messagebox.showwarning("Warning", f"Frame {frame_idx} out of range.")
            return
        
        self.current_frame_idx = frame_idx
        
        # Read the video frame (CSV frames start at 1, video index at 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            # Convert for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Target size
            target_w, target_h = 640, 480

            # Original size
            h, w = frame_rgb.shape[:2]
            ratio = min(target_w / w, target_h / h)
            new_w, new_h = int(w * ratio), int(h * ratio)

            # Resize without distortion
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

            # Create a black canvas and center the resized frame
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized

            img = Image.fromarray(canvas)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.configure(image=imgtk, text="")
            self.video_label.image = imgtk

        
        # Update the frame counter
        frame_number = self.annotations_df.iloc[frame_idx]['Frame']
        self.frame_info_label.config(text=f"Frame: {frame_number} / {len(self.annotations_df)}")
        
        # Show the current annotations
        self.update_current_annotations(frame_idx)
        
        # Update the checkboxes
        self.update_checkboxes(frame_idx)
        
        # Refresh the labels with the shortcuts if needed
        if hasattr(self, 'keyboard_shortcuts') and self.keyboard_shortcuts:
            self.update_checkbox_labels()
    
    def update_current_annotations(self, frame_idx):
        row = self.annotations_df.iloc[frame_idx]
        
        active_behaviors = []
        for behavior in self.behavior_columns:
            if row[behavior] == 1:
                active_behaviors.append(behavior)
        
        self.current_annotation_text.config(state=tk.NORMAL)
        self.current_annotation_text.delete(1.0, tk.END)
        
        if active_behaviors:
            self.current_annotation_text.insert(tk.END, "Active behaviours:\n\n")
            for behavior in active_behaviors:
                self.current_annotation_text.insert(tk.END, f"  ✓ {behavior}\n")
        else:
            self.current_annotation_text.insert(tk.END, "No active behaviour")
        
        self.current_annotation_text.config(state=tk.DISABLED)
    
    def update_checkboxes(self, frame_idx):
        row = self.annotations_df.iloc[frame_idx]
        
        for behavior in self.behavior_columns:
            self.checkboxes[behavior].set(int(row[behavior]))
    
    def next_frame(self):
        if self.current_frame_idx < len(self.annotations_df) - 1:
            self.display_frame(self.current_frame_idx + 1)
        else:
            messagebox.showinfo("Info", "Last frame reached.")
    
    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.display_frame(self.current_frame_idx - 1)
        else:
            messagebox.showinfo("Info", "First frame reached.")
    
    def jump_to_frame(self):
        try:
            frame_num = int(self.jump_entry.get())
            # Row index of this frame number
            frame_idx = self.annotations_df[self.annotations_df['Frame'] == frame_num].index
            if len(frame_idx) > 0:
                self.display_frame(frame_idx[0])
            else:
                messagebox.showerror("Error", f"Frame {frame_num} not found in the annotations.")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid frame number.")
    
    def apply_to_range(self):
        """Apply the checked behaviours to a range of frames."""
        if self.annotations_df is None:
            messagebox.showwarning("Warning", "No annotation loaded.")
            return
        
        try:
            start_frame = int(self.range_start_entry.get())
            end_frame = int(self.range_end_entry.get())
            
            if start_frame > end_frame:
                messagebox.showerror("Error", "The start frame must be lower than or equal to the end frame.")
                return
            
            # Corresponding row indices
            start_idx = self.annotations_df[self.annotations_df['Frame'] == start_frame].index
            end_idx = self.annotations_df[self.annotations_df['Frame'] == end_frame].index
            
            if len(start_idx) == 0 or len(end_idx) == 0:
                messagebox.showerror("Error", "Invalid frame numbers.")
                return
            
            start_idx = start_idx[0]
            end_idx = end_idx[0]
            
            # Checkbox values
            checkbox_values = {}
            for behavior in self.behavior_columns:
                checkbox_values[behavior] = self.checkboxes[behavior].get()
            
            # Apply to every frame of the range
            for idx in range(start_idx, end_idx + 1):
                for behavior in self.behavior_columns:
                    self.annotations_df.at[idx, behavior] = checkbox_values[behavior]
            
            self.modified = True
            
            # Refresh the display
            self.update_current_annotations(self.current_frame_idx)
            
            num_frames = end_idx - start_idx + 1
            messagebox.showinfo("Success", f"Annotations applied to {num_frames} frames ({start_frame} to {end_frame}).")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid frame numbers.")
    
    def confirm_correction(self):
        if self.annotations_df is None:
            return
        
        # Write the checkbox values into the annotations
        for behavior in self.behavior_columns:
            self.annotations_df.at[self.current_frame_idx, behavior] = self.checkboxes[behavior].get()
        
        self.modified = True
        
        # Refresh the display
        self.update_current_annotations(self.current_frame_idx)
        
        
        messagebox.showinfo("Success", f"Annotation of frame {self.annotations_df.iloc[self.current_frame_idx]['Frame']} updated.")
        
        # Go to the next frame
        self.next_frame()
    
    def save_annotations(self):
        if self.annotations_df is None:
            messagebox.showwarning("Warning", "No annotation to save.")
            return
        
        if self.csv_path:
            self.annotations_df.to_csv(self.csv_path, index=False)
            self.modified = False
            messagebox.showinfo("Success", f"Annotations saved to {os.path.basename(self.csv_path)}")
        else:
            self.save_annotations_as()
    
    def save_annotations_as(self):
        if self.annotations_df is None:
            messagebox.showwarning("Warning", "No annotation to save.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save the annotations",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            self.annotations_df.to_csv(filepath, index=False)
            self.csv_path = filepath
            self.modified = False
            messagebox.showinfo("Success", f"Annotations saved to {os.path.basename(filepath)}")
    
    def quit_app(self):
        if self.modified:
            response = messagebox.askyesnocancel("Quit", "Save the changes before quitting?")
            if response is None:  # Cancel
                return
            elif response:  # Yes
                self.save_annotations()
        
        if self.cap is not None:
            self.cap.release()
        self.root.quit()

def main():
    ap = argparse.ArgumentParser(description="Frame-by-frame annotation correction GUI.")
    ap.add_argument("--video", help="Video to open at start-up (optional).")
    ap.add_argument("--csv", help="Annotation CSV to open at start-up (optional).")
    args = ap.parse_args()

    root = tk.Tk()
    app = AnnotationCorrectionTool(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    if args.csv:
        app.load_csv(args.csv)
    if args.video:
        app.load_video(args.video)
    root.mainloop()

if __name__ == "__main__":
    main()
