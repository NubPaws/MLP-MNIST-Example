import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from mlp import MLP

# Choose the best available resampling filter for downsampling
try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS  # Pillow ≥9
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS            # older Pillow

CANVAS_SIZE: int       = 20*20
GRID_SIZE: int         = 28
BRUSH_RADIUS: int      = 8
LABEL_MARGIN: int      = 20
BAR_AREA_HEIGHT: int   = GRID_SIZE * 10
CANVAS_BAR_HEIGHT: int = BAR_AREA_HEIGHT + LABEL_MARGIN


class DrawApp:
    def __init__(self, model_path: str) -> None:
        self.net: MLP = MLP.load(model_path)

        self.root: tk.Tk = tk.Tk()
        self.root.title("Draw & Predict MNIST")

        # drawing image & canvas
        self.image: Image.Image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw: ImageDraw.ImageDraw = ImageDraw.Draw(self.image)
        self.canvas: tk.Canvas = tk.Canvas(
            self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black"
        )
        self.canvas.grid(row=0, column=0, padx=5, pady=5)
        self.canvas_image: ImageTk.PhotoImage = ImageTk.PhotoImage(self.image)
        self.canvas_img_id: int = self.canvas.create_image(
            0, 0, anchor="nw", image=self.canvas_image
        )

        # bar graph canvas
        self.bar_canvas = tk.Canvas(
            self.root,
            width=GRID_SIZE * 10,
            height=CANVAS_BAR_HEIGHT,
            bg="white"
        )
        self.bar_canvas.grid(row=0, column=1, padx=5, pady=5)

        # clear button
        btn: tk.Button = tk.Button(self.root, text="Clear", command=self.clear)
        btn.grid(row=1, column=0, columnspan=2, pady=5)

        # events
        self.canvas.bind("<B1-Motion>", self.paint)
        # update loop
        self.update_bars()
        self.root.mainloop()

    def paint(self, event: tk.Event) -> None:
        x: int = event.x
        y: int = event.y
        self.draw.ellipse(
            [x - BRUSH_RADIUS, y - BRUSH_RADIUS, x + BRUSH_RADIUS, y + BRUSH_RADIUS],
            fill=255,
        )
        self.canvas_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.canvas_img_id, image=self.canvas_image)

    def clear(self) -> None:
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.canvas_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.canvas_img_id, image=self.canvas_image)

    def update_bars(self) -> None:
        # 1. downsample
        small = self.image.resize((GRID_SIZE, GRID_SIZE), RESAMPLE_FILTER)
        arr = np.asarray(small, dtype=np.float32).reshape(1, -1) / 255.0
        probs = self.net.predict_proba(arr).flatten()  # should be length-10

        # 2. clear previous bars
        self.bar_canvas.delete("all")

        # 3. dimensions
        total_width = GRID_SIZE * 10
        num_classes = len(probs)  # should be 10
        w = total_width / num_classes
        max_bar_height = BAR_AREA_HEIGHT - 10  # leave 10px padding above bars

        # 4. draw each bar + label
        for cls_idx, p in enumerate(probs):
            x0 = cls_idx * w + 5
            x1 = x0 + w - 10
            y1 = BAR_AREA_HEIGHT
            y0 = y1 - (p * max_bar_height)
            # bar
            self.bar_canvas.create_rectangle(x0, y0, x1, y1, fill="blue")
            # label: place it in the 20px label margin under the bars
            label_x = x0 + (x1 - x0) / 2
            label_y = BAR_AREA_HEIGHT + 2
            self.bar_canvas.create_text(
                label_x,
                label_y,
                text=str(cls_idx),
                anchor="n"   # north: text top at label_y
            )

        # 5. schedule next update
        self.root.after(50, self.update_bars)


if __name__ == "__main__":
    DrawApp("mlp_mnist.npz")
