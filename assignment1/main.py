import tkinter as tk
import math
from tkinter import messagebox

# Transformation functions using matrix operations
def translate(point, tx, ty):
    return (point[0] + tx, point[1] + ty)

def rotate(point, angle):
    rad = math.radians(angle)
    x = point[0] * math.cos(rad) - point[1] * math.sin(rad)
    y = point[0] * math.sin(rad) + point[1] * math.cos(rad)
    return (x, y)

def scale(point, sx, sy):
    return (point[0] * sx, point[1] * sy)

def transform_points(points, tx, ty, angle, sx, sy):
    transformed_points = []
    for point in points:
        # Apply translation if tx or ty is provided
        if tx is not None or ty is not None:
            tx_val = tx if tx is not None else 0
            ty_val = ty if ty is not None else 0
            point = translate(point, tx_val, ty_val)
        # Apply rotation if angle is provided
        if angle is not None:
            point = rotate(point, angle)
        # Apply scaling if sx or sy is provided
        if sx is not None or sy is not None:
            sx_val = sx if sx is not None else 1
            sy_val = sy if sy is not None else 1
            point = scale(point, sx_val, sy_val)
        transformed_points.append(point)
    return transformed_points

# Function to draw a grid with coordinates
def draw_grid(canvas, width, height, spacing=50):
    # Draw vertical lines
    for x in range(0, width, spacing):
        canvas.create_line(x, 0, x, height, fill="black")
        canvas.create_text(x + 5, 0, text=str(x), fill="black", anchor=tk.NW)
    # Draw horizontal lines
    for y in range(0, height, spacing):
        if y == 0:
            continue
        canvas.create_line(0, y, width, y, fill="black")
        canvas.create_text(5, y + 5, text=str(y), fill="black", anchor=tk.NW)

# GUI Application
class TransformApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Transformations")
        self.root.resizable(False, False)
        self.point_radius = 4

        # Left panel for placing points
        self.left_panel = tk.Canvas(root, width=300, height=400, bg="white")
        self.left_panel.grid(row=0, column=0, padx=10, pady=10)
        self.left_panel.bind("<Button-1>", self.place_point)
        draw_grid(self.left_panel, 300, 400)  # Draw grid on the left panel

        # Right panel for displaying transformed points
        self.right_panel = tk.Canvas(root, width=300, height=400, bg="lightgray")
        self.right_panel.grid(row=0, column=1, padx=10, pady=10)
        draw_grid(self.right_panel, 300, 400)  # Draw grid on the right panel

        # Input field for X translation
        self.tx_label = tk.Label(root, text="Translate X (tx):")
        self.tx_label.grid(row=1, column=0, padx=5, pady=5)
        self.tx_entry = tk.Entry(root)
        self.tx_entry.grid(row=1, column=1, padx=5, pady=5)

        # Input field for Y translation
        self.ty_label = tk.Label(root, text="Translate Y (ty):")
        self.ty_label.grid(row=2, column=0, padx=5, pady=5)
        self.ty_entry = tk.Entry(root)
        self.ty_entry.grid(row=2, column=1, padx=5, pady=5)

        # Input fields for rotation
        self.angle_label = tk.Label(root, text="Rotation Angle (degrees):")
        self.angle_label.grid(row=3, column=0, padx=5, pady=5)
        self.angle_entry = tk.Entry(root)
        self.angle_entry.grid(row=3, column=1, padx=5, pady=5)

        # Input fields for scaling X
        self.sx_label = tk.Label(root, text="Scale X (sx):")
        self.sx_label.grid(row=4, column=0, padx=5, pady=5)
        self.sx_entry = tk.Entry(root)
        self.sx_entry.grid(row=4, column=1, padx=5, pady=5)

        # Input fields for scaling Y
        self.sy_label = tk.Label(root, text="Scale Y (sy):")
        self.sy_label.grid(row=5, column=0, padx=5, pady=5)
        self.sy_entry = tk.Entry(root)
        self.sy_entry.grid(row=5, column=1, padx=5, pady=5)

        # Transform button
        self.transform_button = tk.Button(root, text="Transform", command=self.transform)
        self.transform_button.grid(row=6, column=0, columnspan=2, pady=10)

        # Store original and transformed points
        self.original_points = []
        self.transformed_points = []

        # Labels for displaying point coordinates in the top-right
        self.original_points_label = tk.Label(root, text="Original Points: ", anchor="w")
        self.original_points_label.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

        self.transformed_points_label = tk.Label(root, text="Transformed Points: ", anchor="w")
        self.transformed_points_label.grid(row=8, column=0, columnspan=2, padx=5, pady=5)

        # Clear button
        self.clear_button = tk.Button(root, text="Clear Points", command=self.clear_points)
        self.clear_button.grid(row=9, column=0, columnspan=2, pady=10)



    # Event handler for placing points on the left panel
    def place_point(self, event):
        x, y = event.x, event.y
        self.original_points.append((x, y))
        self.left_panel.create_oval(x-self.point_radius, y-self.point_radius, x+self.point_radius, y+self.point_radius, fill="black")
        self.update_original_points_label()

    # Update the original points label with coordinates
    def update_original_points_label(self):
        label_text = "Original Points: "
        for point in self.original_points:
            label_text += f"({point[0]}, {point[1]})  |  "
        self.original_points_label.config(text=label_text)

    # Update the transformed points label with coordinates
    def update_transformed_points_label(self):
        label_text = "Transformed Points: "
        for point in self.transformed_points:
            label_text += f"({point[0]:.2f}, {point[1]:.2f})  |  "
        self.transformed_points_label.config(text=label_text)

    # Event handler for the Transform button
    def transform(self):
        try:
            tx = float(self.tx_entry.get()) if self.tx_entry.get() else 0
            ty = float(self.ty_entry.get()) if self.ty_entry.get() else 0
            angle = float(self.angle_entry.get()) if self.angle_entry.get() else 0
            sx = float(self.sx_entry.get()) if self.sx_entry.get() else 1
            sy = float(self.sy_entry.get()) if self.sy_entry.get() else 1
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers in the fields.")
            return
        # Apply transformations
        self.transformed_points = transform_points(self.original_points, tx, ty, angle, sx, sy)

        # Clear right panel and display transformed points
        self.right_panel.delete("all")
        draw_grid(self.right_panel, 300, 400)  # Redraw grid on the right panel
        for point in self.transformed_points:
            x, y = point
            self.right_panel.create_oval(x-self.point_radius, y-self.point_radius, x+self.point_radius, y+self.point_radius, fill="blue")

        # Update the transformed points label
        self.update_transformed_points_label()

    def clear_points(self):
        self.original_points = []
        self.transformed_points = []
        self.left_panel.delete("all")
        self.right_panel.delete("all")
        draw_grid(self.left_panel, 300, 400)
        draw_grid(self.right_panel, 300, 400)
        self.update_original_points_label()
        self.update_transformed_points_label()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = TransformApp(root)
    root.mainloop()
