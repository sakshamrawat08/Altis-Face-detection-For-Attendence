# admin_panel_enhanced.py
# Colorful Admin Panel with integrated camera preview (Tkinter)
# Requirements: opencv-contrib-python, Pillow, pandas, numpy
# Place haarcascade_frontalface_default.xml in the same folder.

import os
import cv2
import json
import threading
import datetime
import shutil
import pandas as pd
import numpy as np
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# ---------------- Config ----------------
DATASET = "dataset"
TRAINER = "trainer"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
ATTEND_FILE = "attendance.csv"
IMG_SIZE = (200, 200)
CAPTURE_COUNT = 30
CONF_THRESHOLD = 70
CONSECUTIVE_REQUIRED = 3
MIN_FACE = 60
DETECTION_SCALE = 0.5
FRAME_SKIP = 2

# Save color thumbnail as well? (True = save color image alongside grayscale)
SAVE_COLOR_THUMBNAIL = True

os.makedirs(DATASET, exist_ok=True)
os.makedirs(TRAINER, exist_ok=True)

def sanitize(name):
    return name.strip().replace(" ", "_")

def ensure_attendance_file():
    if not os.path.exists(ATTEND_FILE):
        pd.DataFrame(columns=["name", "date", "time"]).to_csv(ATTEND_FILE, index=False)
ensure_attendance_file()

# ---------------- Theme / Colors ----------------
ACCENT = "#3366FF"
ACCENT_DARK = "#2A50CC"
BG = "#F4F7FB"
PANEL = "#FFFFFF"
SIDEBAR = "#E9F0FF"
LOG_BG = "#0F1724"
LOG_FG = "#E6EEF5"
BTN_FG = "#FFFFFF"

FONT_MAIN = ("Inter", 11)
FONT_BOLD = ("Inter", 12, "bold")
SMALL_FONT = ("Inter", 10)

def make_styles():
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    style.configure("Accent.TButton",
                    background=ACCENT, foreground=BTN_FG, font=FONT_MAIN,
                    padding=6)
    style.map("Accent.TButton",
              background=[('active', ACCENT_DARK)])
    style.configure("Secondary.TButton",
                    background="#F0F4FF", foreground="#0B1220", font=FONT_MAIN,
                    padding=6)
    style.configure("Accent.Horizontal.TProgressbar",
                    troughcolor="#E6EEFF", background=ACCENT)
    style.configure("TLabel", background=BG)
    style.configure("TFrame", background=BG)

# ---------------- VideoPreview (integrated) ----------------
class VideoPreview(Toplevel):
    def __init__(self, parent, mode="preview", register_name=None, on_close=None):
        super().__init__(parent)
        self.parent = parent
        self.mode = mode
        self.register_name = register_name
        self.on_close = on_close
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.title(f"Camera - {mode.capitalize()}")
        self.resizable(False, False)

        # Image label
        self.img_label = Label(self, bg="black")
        self.img_label.pack(padx=8, pady=8)
        self.img_label.config(width=640, height=480)
        try:
            self.geometry("700x620")
        except:
            pass

        # control frame
        ctrl = Frame(self, bg=BG)
        ctrl.pack(fill=X, padx=8, pady=(0,8))
        if mode == "register":
            self.capture_btn = ttk.Button(ctrl, text="Capture Face", style="Accent.TButton", command=self.capture_face)
            self.capture_btn.pack(side=LEFT, padx=6)
            self.count_label = Label(ctrl, text="Saved: 0", bg=BG, font=SMALL_FONT)
            self.count_label.pack(side=LEFT, padx=8)
            # prepare save folder
            self.save_folder = os.path.join(DATASET, sanitize(self.register_name))
            os.makedirs(self.save_folder, exist_ok=True)
            # count only grayscale training files (ignore color thumbs)
            try:
                self.saved_count = len([f for f in os.listdir(self.save_folder) if f.lower().endswith(('.jpg','.png')) and "_color" not in f])
            except FileNotFoundError:
                self.saved_count = 0
            self.count_label.config(text=f"Saved: {self.saved_count}")
        if mode == "attendance":
            self.status_label = Label(ctrl, text="Status: Running", bg=BG, font=SMALL_FONT)
            self.status_label.pack(side=LEFT, padx=6)

        # load cascade
        if not os.path.exists(CASCADE_FILE):
            messagebox.showerror("Missing file", f"'{CASCADE_FILE}' not found in project folder.")
            self.destroy()
            return
        self.cascade = cv2.CascadeClassifier(CASCADE_FILE)

        # open camera and set resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            messagebox.showerror("Camera", "Cannot open camera. Grant permission and close other camera apps.")
            self.destroy()
            return

        # attendance model
        self.recognizer = None
        self.labels = {}
        if mode == "attendance":
            trainer_file = os.path.join(TRAINER, "trainer.yml")
            labels_file = os.path.join(TRAINER, "labels.json")
            if not os.path.exists(trainer_file) or not os.path.exists(labels_file):
                messagebox.showerror("Model", "Train model first (use Train Model button).")
                self.cap.release(); self.destroy(); return
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(trainer_file)
                with open(labels_file, "r") as f:
                    self.labels = {int(k): v for k, v in json.load(f).items()}
            except Exception as e:
                messagebox.showerror("Model load", f"Failed to load model: {e}")
                self.cap.release(); self.destroy(); return
            self.pred_queue = []

        # loop control
        self.running = True
        self.frame_count = 0
        self.after(10, self._update_frame)

    def _update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.after(30, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count = (self.frame_count + 1) % FRAME_SKIP
        faces = []
        if self.frame_count == 0:
            try:
                small = cv2.resize(gray, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
                small_faces = self.cascade.detectMultiScale(small, scaleFactor=1.2, minNeighbors=5)
                for (sx, sy, sw, sh) in small_faces:
                    x = int(sx / DETECTION_SCALE)
                    y = int(sy / DETECTION_SCALE)
                    w = int(sw / DETECTION_SCALE)
                    h = int(sh / DETECTION_SCALE)
                    faces.append((x, y, w, h))
            except Exception:
                faces = []

        if faces:
            for (x, y, w, h) in faces:
                if w < MIN_FACE or h < MIN_FACE:
                    continue
                cv2.rectangle(display, (x, y), (x + w, y + h), (51, 102, 255), 2)
                if self.mode == "attendance" and self.recognizer is not None:
                    try:
                        face = gray[y:y + h, x:x + w]
                        f_rs = cv2.resize(face, IMG_SIZE)
                        f_eq = cv2.equalizeHist(f_rs)
                        label, conf = self.recognizer.predict(f_eq)
                        name = self.labels.get(label, "Unknown").replace("_", " ")
                        cv2.putText(display, f"{name} {int(conf)}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (51, 102, 255), 2)
                        self.pred_queue.append((label, conf, name))
                        if len(self.pred_queue) > CONSECUTIVE_REQUIRED:
                            self.pred_queue.pop(0)
                        if len(self.pred_queue) == CONSECUTIVE_REQUIRED:
                            labs = [p[0] for p in self.pred_queue]
                            confs = [p[1] for p in self.pred_queue]
                            if len(set(labs)) == 1 and (sum(confs) / len(confs)) < CONF_THRESHOLD:
                                recog_name = self.labels.get(labs[0], "Unknown").replace("_", " ")
                                self._mark_attendance(recog_name)
                            self.pred_queue.clear()
                    except Exception:
                        pass

        try:
            disp = cv2.resize(display, (640, 480))
        except Exception:
            disp = display
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=pil)
        self.img_label.imgtk = imgtk
        self.img_label.configure(image=imgtk)
        self.after(10, self._update_frame)

    def capture_face(self):
        """Capture colored preview but save both grayscale(face) and color thumbnail.
           Robustly update saved_count and count_label after each save.
        """
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Camera", "Failed to read frame")
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect on small image then scale up
        try:
            small = cv2.resize(gray, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            sfaces = self.cascade.detectMultiScale(small, scaleFactor=1.2, minNeighbors=5)
        except Exception:
            sfaces = []

        if len(sfaces) == 0:
            messagebox.showinfo("No face", "No face detected. Align face and try again.")
            return

        sx, sy, sw, sh = sfaces[0]
        x = int(sx / DETECTION_SCALE); y = int(sy / DETECTION_SCALE)
        w = int(sw / DETECTION_SCALE); h = int(sh / DETECTION_SCALE)

        if w < MIN_FACE or h < MIN_FACE:
            messagebox.showinfo("Small face", "Face too small. Move closer to camera.")
            return

        # prepare grayscale face for training
        face_gray = gray[y:y + h, x:x + w]
        try:
            face_gray = cv2.resize(face_gray, IMG_SIZE)
        except:
            pass

        # prepare color thumbnail for demo (resize to IMG_SIZE)
        if SAVE_COLOR_THUMBNAIL:
            face_color = frame[y:y + h, x:x + w]
            try:
                face_color = cv2.resize(face_color, IMG_SIZE)
            except:
                pass

        # ensure folder exists (just in case)
        try:
            os.makedirs(self.save_folder, exist_ok=True)
        except Exception as e:
            self.parent.log(f"Error creating folder: {e}")

        # compute index by counting only grayscale training files (ignore *_color)
        try:
            existing = [f for f in os.listdir(self.save_folder)
                        if f.lower().endswith(('.jpg', '.png')) and "_color" not in f]
        except FileNotFoundError:
            existing = []
        idx = len(existing)
        base_name = sanitize(self.register_name)

        # filenames
        gray_fname = os.path.join(self.save_folder, f"{base_name}_{idx}.jpg")
        cv2.imwrite(gray_fname, face_gray)

        if SAVE_COLOR_THUMBNAIL:
            color_fname = os.path.join(self.save_folder, f"{base_name}_{idx}_color.jpg")
            cv2.imwrite(color_fname, face_color)
            self.parent.log(f"Saved color thumbnail: {color_fname}")

        # Recalculate saved_count (robust) and update label
        try:
            new_existing = [f for f in os.listdir(self.save_folder)
                            if f.lower().endswith(('.jpg', '.png')) and "_color" not in f]
            self.saved_count = len(new_existing)
        except Exception:
            # fallback increment if list failed
            self.saved_count = idx + 1

        if hasattr(self, "count_label") and self.count_label:
            try:
                self.count_label.config(text=f"Saved: {self.saved_count}")
            except Exception:
                pass

        self.parent.log(f"Saved grayscale training face → {gray_fname}")
        print(f"[capture] saved: {gray_fname} (count now {self.saved_count})")

    def _mark_attendance(self, name):
        if name == "Unknown":
            return
        att_path = os.path.abspath(ATTEND_FILE)
        if not os.path.exists(att_path) or os.path.getsize(att_path) == 0:
            pd.DataFrame(columns=["name", "date", "time"]).to_csv(att_path, index=False)
        try:
            df = pd.read_csv(att_path)
        except Exception:
            df = pd.DataFrame(columns=["name", "date", "time"])
        today = datetime.date.today().isoformat()
        if not ((df["name"] == name) & (df["date"] == today)).any():
            now = datetime.datetime.now().strftime("%H:%M:%S")
            new = pd.DataFrame([{"name": name, "date": today, "time": now}])
            df = pd.concat([df, new], ignore_index=True)
            df.to_csv(att_path, index=False)
            self.parent.log(f"Marked attendance: {name} (written to {att_path})")
            print(f"Marked attendance: {name} -> {att_path}")

    def _on_close(self):
        self.running = False
        try:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
        except:
            pass
        if callable(self.on_close):
            try:
                self.on_close()
            except:
                pass
        self.destroy()

# ---------------- Main Admin App ----------------
class AdminApp:
    def __init__(self, root):
        make_styles()
        self.root = root
        root.title("FaceAttendance - Admin (Colorful UI)")
        root.configure(bg=BG)
        root.geometry("1000x640")

        # header
        header = Frame(root, bg=ACCENT, height=64)
        header.pack(fill=X)
        title = Label(header, text="FaceAttendance — Admin Panel", bg=ACCENT, fg="white", font=("Inter", 16, "bold"))
        title.pack(side=LEFT, padx=16, pady=12)
        sub = Label(header, text="Register · Train · Take Attendance", bg=ACCENT, fg="#EAF0FF", font=SMALL_FONT)
        sub.pack(side=LEFT, padx=8, pady=12)

        main = Frame(root, bg=BG)
        main.pack(fill=BOTH, expand=True, padx=12, pady=12)

        # left pane (users)
        left = Frame(main, bg=SIDEBAR, width=320)
        left.pack(side=LEFT, fill=Y, padx=(0,10), pady=4)
        Label(left, text="Registered Users", bg=SIDEBAR, fg="#07213A", font=FONT_BOLD).pack(pady=(10,4))
        self.user_listbox = Listbox(left, width=32, height=18, bd=0, bg=PANEL, fg="#07213A", font=FONT_MAIN, highlightthickness=0, selectbackground=ACCENT)
        self.user_listbox.pack(pady=6, padx=8)
        self.thumb_canvas = Canvas(left, width=280, height=200, bg=SIDEBAR, bd=0, highlightthickness=0)
        self.thumb_canvas.pack(pady=6)
        self.refresh_users()

        # center pane (controls)
        center = Frame(main, bg=BG)
        center.pack(side=LEFT, fill=BOTH, expand=True)
        Label(center, text="Controls", bg=BG, fg="#07213A", font=FONT_BOLD).pack(anchor='w')
        ctrl = Frame(center, bg=BG); ctrl.pack(fill=X, pady=8)
        self.name_entry = Entry(ctrl, width=36, font=FONT_MAIN)
        self.name_entry.grid(row=0, column=0, padx=6)
        self.name_entry.insert(0, "Enter full name and press Register")
        reg_btn = ttk.Button(ctrl, text="Register / Capture", style="Accent.TButton", command=self.on_register)
        reg_btn.grid(row=0, column=1, padx=6)
        train_btn = ttk.Button(center, text="Train Model", style="Secondary.TButton", command=self.train_thread)
        train_btn.pack(pady=(10,6))
        att_btn = ttk.Button(center, text="Start Attendance", style="Accent.TButton", command=self.start_attendance)
        att_btn.pack(pady=6)

        self.progress = ttk.Progressbar(center, length=560, mode='determinate', style="Accent.Horizontal.TProgressbar")
        self.progress.pack(pady=6)
        # log area with dark theme
        log_frame = Frame(center, bg=LOG_BG)
        log_frame.pack(fill=BOTH, expand=True, pady=8)
        self.log_text = Text(log_frame, height=14, bg=LOG_BG, fg=LOG_FG, bd=0, insertbackground=LOG_FG, font=("Consolas",10))
        self.log_text.pack(fill=BOTH, expand=True, padx=6, pady=6)

        # bottom actions
        bottom = Frame(root, bg=BG); bottom.pack(side=BOTTOM, fill=X, pady=(0,8))
        view_btn = ttk.Button(bottom, text="View Attendance CSV", style="Secondary.TButton", command=self.view_csv)
        view_btn.pack(side=LEFT, padx=8)
        rfr_btn = ttk.Button(bottom, text="Refresh Users", style="Secondary.TButton", command=self.refresh_users)
        rfr_btn.pack(side=LEFT, padx=8)
        del_btn = ttk.Button(bottom, text="Delete Selected User", style="Secondary.TButton", command=self.delete_user)
        del_btn.pack(side=LEFT, padx=8)

    def log(self, msg):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(END, f"[{t}] {msg}\n")
        self.log_text.see(END)

    def refresh_users(self):
        self.user_listbox.delete(0, END)
        users = sorted([d for d in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET, d))])
        for u in users:
            self.user_listbox.insert(END, u)
        # thumbnails
        self.thumb_canvas.delete("all")
        self.thumb_canvas.images = []
        x, y = 10, 10
        for u in users[:6]:
            folder = os.path.join(DATASET, u)
            imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
            if not imgs:
                continue
            p = os.path.join(folder, imgs[0])
            try:
                img = Image.open(p).resize((80, 80))
                tkimg = ImageTk.PhotoImage(img)
                self.thumb_canvas.create_rectangle(x-4, y-4, x+84, y+104, fill=PANEL, outline="")
                self.thumb_canvas.create_image(x, y, anchor='nw', image=tkimg)
                self.thumb_canvas.create_text(x + 40, y + 92, text=u.replace("_", " "), anchor='n', width=90, font=SMALL_FONT)
                self.thumb_canvas.images.append(tkimg)
                x += 90
            except Exception as e:
                self.log(f"Thumb error: {e}")

    def on_register(self):
        name = self.name_entry.get().strip()
        if not name or name.lower().startswith("enter"):
            messagebox.showwarning("Input", "Please type a valid name")
            return
        self.preview = VideoPreview(self.root, mode="register", register_name=name, on_close=self.refresh_users)
        self.log(f"Registering: {name} — use Capture button to save faces")

    def train_thread(self):
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        self.log("Training started...")
        faces, labels = [], []
        label_map = {}
        idx = 0
        for person in sorted(os.listdir(DATASET)):
            pdir = os.path.join(DATASET, person)
            if not os.path.isdir(pdir):
                continue
            label_map[idx] = person
            for f in sorted(os.listdir(pdir)):
                p = os.path.join(pdir, f)
                # skip color thumbnails during training
                if "_color" in p:
                    continue
                im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                try:
                    im = cv2.resize(im, IMG_SIZE)
                except:
                    pass
                im = cv2.equalizeHist(im)
                faces.append(im)
                labels.append(idx)
            idx += 1
        if len(faces) == 0:
            self.log("No faces found. Capture images first.")
            return
        try:
            rec = cv2.face.LBPHFaceRecognizer_create()
            rec.train(faces, np.array(labels, dtype=np.int32))
            os.makedirs(TRAINER, exist_ok=True)
            rec.write(os.path.join(TRAINER, "trainer.yml"))
            with open(os.path.join(TRAINER, "labels.json"), "w") as f:
                json.dump(label_map, f)
            self.log(f"Training completed. Labels: {label_map}")
        except Exception as e:
            self.log(f"Training failed: {e}")

    def start_attendance(self):
        self.preview = VideoPreview(self.root, mode="attendance", on_close=lambda: self.log("Attendance closed"))
        self.log("Attendance window opened")

    def view_csv(self):
        if os.path.exists(ATTEND_FILE):
            try:
                os.system(f'open "{os.path.abspath(ATTEND_FILE)}"')
            except:
                messagebox.showinfo("File", f"attendance file: {os.path.abspath(ATTEND_FILE)}")
        else:
            messagebox.showinfo("No file", "No attendance file yet.")

    def delete_user(self):
        try:
            sel = self.user_listbox.get(self.user_listbox.curselection())
        except:
            messagebox.showwarning("Delete", "Select a user from the list.")
            return
        if messagebox.askyesno("Confirm", f"Delete {sel}?"):
            shutil.rmtree(os.path.join(DATASET, sel))
            self.log(f"Deleted user {sel}")
            self.refresh_users()

# ---------------- Run App ----------------
if __name__ == "__main__":
    root = Tk()
    make_styles()
    app = AdminApp(root)
    root.mainloop()
