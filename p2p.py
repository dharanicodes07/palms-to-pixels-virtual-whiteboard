# air_draw_toolbar.py
import cv2
import mediapipe as mp
import time
import numpy as np
from tkinter import Tk, filedialog

class HandDetector:
    def __init__(self, detectionCon=0.5, maxHands=2):
        self.detectionCon = detectionCon
        self.maxHands = maxHands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        # returns list of 5 values (thumb..pinky) 1 if up else 0
        fingers = []
        if len(self.lmList) == 0:
            return [0,0,0,0,0]
        # Thumb (x coordinate check for mirrored image)
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers by y coordinate
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def upload_background():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Background Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        bg = cv2.imread(file_path)
        if bg is None:
            return None
        bg = cv2.resize(bg, (640, 480))
        return bg
    return None


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    detector = HandDetector(detectionCon=0.8)
    pTime = 0

    # Canvas and background
    canvas = np.zeros((480, 640, 3), np.uint8)
    bg_img = np.zeros((480, 640, 3), np.uint8)  # default plain black background

    # Toolbar configuration
    toolbar_h = 70
    btn_w = 90
    btn_h = toolbar_h - 10
    btn_y = 5
    # Buttons left to right: Red, Green, Blue, Yellow, Black, White, Eraser, Upload
    buttons = [
        {"name": "Red",    "color": (0,0,255)},
        {"name": "Green",  "color": (0,255,0)},
        {"name": "Blue",   "color": (255,0,0)},
        {"name": "Yellow", "color": (0,255,255)},
        {"name": "Black",  "color": (0,0,0)},
        {"name": "White",  "color": (255,255,255)},
        {"name": "Eraser", "color": None},
        {"name": "Upload", "color": None},
    ]

    # Pen settings
    pen_color = (255, 0, 255)  # default magenta-ish
    brush_thickness = 6
    eraser_thickness = 50
    erase_mode = False

    xp, yp = 0, 0

    # Debounce for selection: allow one selection every 0.6s
    last_select_time = 0
    select_cooldown = 0.6

    print("Selection mode: index+middle finger up -> touch a top button to select.")
    print("Drawing mode: index finger up only -> draw on screen.")
    print("Buttons: Red, Green, Blue, Yellow, Black, White, Eraser, Upload")

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)  # mirror for natural interaction

        # Draw toolbar background
        toolbar = np.zeros((toolbar_h, 640, 3), np.uint8) + 220  # light gray
        # Draw buttons
        for i, btn in enumerate(buttons):
            x1 = i * btn_w + 5
            x2 = x1 + btn_w - 10
            # Button rectangle
            cv2.rectangle(toolbar, (x1, btn_y), (x2, btn_y + btn_h), (200,200,200), -1)
            # If it's a color button, show the color
            if btn["name"] in ["Red","Green","Blue","Yellow","Black","White"]:
                color = btn["color"]
                cv2.rectangle(toolbar, (x1+8, btn_y+8), (x2-8, btn_y+btn_h-8), color, -1)
                cv2.putText(toolbar, btn["name"], (x1+6, btn_y+btn_h-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) if btn["name"]!="Black" else (255,255,255), 1)
            else:
                # Eraser or Upload label
                cv2.putText(toolbar, btn["name"], (x1+12, btn_y+btn_h//2+6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2)

        # Place toolbar into main image top
        img[0:toolbar_h, 0:640] = toolbar

        # Add background image blended with camera feed (slightly visible)
        if bg_img is not None:
            cam_small = cv2.resize(img, (640, 480))
            mixed = cv2.addWeighted(cam_small, 0.4, bg_img, 0.6, 0)
            # Overwrite below toolbar area only (keep toolbar on top)
            img[toolbar_h:480, 0:640] = mixed[toolbar_h:480, 0:640]

        # Process hand
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        # If no landmarks, reset prev positions so lines don't jump
        if len(lmList) == 0:
            xp, yp = 0, 0

        # Get fingers status
        fingers = detector.fingersUp() if len(lmList) else [0,0,0,0,0]

        # Selection mode: index + middle finger up
        if fingers[1] == 1 and fingers[2] == 1:
            xp, yp = 0, 0  # stop drawing while selecting
            x1, y1 = lmList[8][1], lmList[8][2]  # index tip
            # show a small circle where index tip is
            cv2.circle(img, (x1, y1), 10, (255, 100, 0), cv2.FILLED)
            # Check if touching toolbar area
            if y1 < toolbar_h:
                cur_time = time.time()
                if cur_time - last_select_time > select_cooldown:
                    last_select_time = cur_time
                    # Which button?
                    btn_index = x1 // btn_w
                    if btn_index < len(buttons):
                        sel = buttons[int(btn_index)]["name"]
                        if sel in ["Red","Green","Blue","Yellow","Black","White"]:
                            pen_color = buttons[int(btn_index)]["color"]
                            erase_mode = False
                            print(f"Selected color: {sel}")
                        elif sel == "Eraser":
                            erase_mode = True
                            print("Selected: Eraser")
                        elif sel == "Upload":
                            print("Opening file dialog to upload background image...")
                            new_bg = upload_background()
                            if new_bg is not None:
                                bg_img = new_bg
                                print("Background updated.")
                            else:
                                print("No image selected or couldn't load.")
        # Drawing mode: only index finger up (middle finger down)
        elif fingers[1] == 1 and fingers[2] == 0:
            x1, y1 = lmList[8][1], lmList[8][2]
            cv2.circle(img, (x1, y1), 8, pen_color if not erase_mode else (0,0,0), cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            # Draw on canvas
            if erase_mode:
                cv2.line(canvas, (xp, yp), (x1, y1), (0,0,0), eraser_thickness)
            else:
                cv2.line(canvas, (xp, yp), (x1, y1), pen_color, brush_thickness)
            xp, yp = x1, y1
        else:
            xp, yp = 0, 0  # reset when no drawing

        # Merge canvas with image (so drawing overlays camera/background)
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        # keep toolbar area from being affected: copy toolbar back after combination
        full_img = img.copy()
        # Combine only below toolbar
        area_combined = cv2.bitwise_and(full_img[toolbar_h:480, 0:640], imgInv[toolbar_h:480, 0:640])
        area_combined = cv2.bitwise_or(area_combined, canvas[toolbar_h:480, 0:640])
        full_img[toolbar_h:480, 0:640] = area_combined

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime
        cv2.putText(full_img, f"FPS: {int(fps)}", (10, 470),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

        # Show current mode & color label
        mode_text = "Eraser" if erase_mode else "Draw"
        color_text = "White" if pen_color == (255,255,255) else ("Black" if pen_color==(0,0,0) else "")
        # try name deduction for display
        if color_text == "":
            if pen_color == (0,0,255): color_text = "Red"
            elif pen_color == (0,255,0): color_text = "Green"
            elif pen_color == (255,0,0): color_text = "Blue"
            elif pen_color == (0,255,255): color_text = "Yellow"
            else: color_text = str(pen_color)
        cv2.putText(full_img, f"Mode: {mode_text} | Color: {color_text}", (160, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2)

        cv2.imshow("Air Drawing with Toolbar", full_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # save whole view+canvas
            cv2.imwrite("drawing_result.png", full_img)
            print("Saved drawing_result.png")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
