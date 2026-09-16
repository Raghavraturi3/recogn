import cv2
import os
import shutil
from tkinter import Tk, Label, Button, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


def sanitize_name(name):
    name = name.strip()
    name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-"))
    return name.replace(" ", "_")


def save_to_person_folder(image_path, name):
    folder = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    next_index = len(existing) + 1
    dest = os.path.join(folder, f"img_{next_index}.jpg")
    shutil.copy(image_path, dest)
    return dest


class ReferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Add Known Person")

        self.preview_label = Label(root, text="No photo yet", width=40, height=15)
        self.preview_label.pack(pady=10)

        self.status_label = Label(root, text="", fg="green")
        self.status_label.pack()

        Button(root, text="Upload Photo from Folder", command=self.upload_photo, width=30).pack(pady=5)
        Button(root, text="Take Photo with Webcam", command=self.capture_photo, width=30).pack(pady=5)
        Label(root, text="Tip: click again to add more photos\n(same person for different angles,\nor a new person with a new name)",
              fg="gray").pack(pady=10)

    def show_preview(self, path):
        img = Image.open(path)
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        self.preview_label.config(image=photo, text="")
        self.preview_label.image = photo

    def ask_name_and_save(self, image_path):
        name = simpledialog.askstring("Person's Name", "Who is this? (name for this person)")
        if not name:
            messagebox.showwarning("Cancelled", "No name entered, photo not saved.")
            return
        name = sanitize_name(name)
        if not name:
            messagebox.showwarning("Invalid name", "Please enter a valid name.")
            return
        dest = save_to_person_folder(image_path, name)
        self.status_label.config(text=f"Saved to known_faces/{name}/ ({os.path.basename(dest)})")

    def upload_photo(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.show_preview(path)
            self.ask_name_and_save(path)

    def capture_photo(self):
        cap = cv2.VideoCapture(0)
        messagebox.showinfo("Webcam", "A window will open. Press SPACE to capture, ESC to cancel.")
        temp_path = os.path.join(KNOWN_FACES_DIR, "_temp_capture.jpg")
        captured = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Press SPACE to capture", frame)
            key = cv2.waitKey(1)
            if key % 256 == 27:  # ESC
                break
            elif key % 256 == 32:  # SPACE
                cv2.imwrite(temp_path, frame)
                captured = True
                break
        cap.release()
        cv2.destroyAllWindows()
        if captured:
            self.show_preview(temp_path)
            self.ask_name_and_save(temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    root = Tk()
    app = ReferenceApp(root)
    root.mainloop()