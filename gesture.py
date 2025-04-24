import cv2
import mediapipe as mp
import pyttsx3
import time
import sys

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Gesture-to-sentence mapping
gesture_mappings = {
    "OPEN_PALM": "Hello, how are you?",
    "FIST": "I need help, can you assist me?",
    "THUMBS_UP": "Yes, that sounds great!",
    "THUMBS_DOWN": "No, I don't agree.",
    "VICTORY": "Thank you very much!",
    "POINTING_UP": "What is your name?",
    "FLAT_HAND_FORWARD": "Please stop, I need a break.",
    "PALM_DOWN": "I am feeling okay.",
    "INDEX_FINGER_FORWARD": "Can you repeat that?",
    "HAND_WAVE": "Goodbye! Have a nice day."
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

prev_text = ""
recognized_sentence = []
last_gesture_time = time.time()
gesture_timeout = 2.0  # Timeout before finalizing a phrase

# Function to detect predefined hand gestures
def detect_hand_gesture(landmarks):
    """Detects hand gestures based on finger positions."""
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y

    if index_tip < middle_tip < ring_tip < pinky_tip:  
        return "OPEN_PALM"  # ✋ Hello
    elif thumb_tip > index_tip > middle_tip > ring_tip > pinky_tip:  
        return "FIST"  # ✊ I need help
    elif thumb_tip < index_tip and middle_tip < ring_tip < pinky_tip:  
        return "THUMBS_UP"  # 👍 Yes
    elif thumb_tip > index_tip and middle_tip > ring_tip > pinky_tip:  
        return "THUMBS_DOWN"  # 👎 No
    elif index_tip < middle_tip and ring_tip < pinky_tip:  
        return "VICTORY"  # ✌️ Thank you
    elif index_tip < thumb_tip and middle_tip > ring_tip > pinky_tip:  
        return "POINTING_UP"  # ☝️ What is your name?
    elif thumb_tip > index_tip > middle_tip > ring_tip > pinky_tip and pinky_tip > ring_tip:
        return "FLAT_HAND_FORWARD"  # 🖐️ Stop
    elif thumb_tip < index_tip and middle_tip > ring_tip and pinky_tip < ring_tip:
        return "PALM_DOWN"  # 🤚 I am feeling okay
    elif index_tip < middle_tip and ring_tip > pinky_tip and thumb_tip > index_tip:
        return "INDEX_FINGER_FORWARD"  # 👉 Can you repeat that?
    elif index_tip < middle_tip and ring_tip > pinky_tip and thumb_tip < index_tip:
        return "HAND_WAVE"  # 👋 Goodbye
    return None

# Function to convert gestures into conversational sentences and speak them
def recognize_and_speak(gesture):
    global prev_text, recognized_sentence
    if gesture in gesture_mappings:
        recognized_sentence.append(gesture_mappings[gesture])
        print("\n📜 Recognized Phrase:", " ".join(recognized_sentence))

# Function to finalize and speak the sentence
def finalize_and_speak():
    global prev_text, recognized_sentence
    if recognized_sentence:
        final_sentence = " ".join(recognized_sentence)
        if final_sentence != prev_text:
            print("\n🗣️ Speaking:", final_sentence)
            engine.say(final_sentence)
            engine.runAndWait()
            prev_text = final_sentence
        recognized_sentence = []  # Reset after speaking

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            gesture = detect_hand_gesture(landmarks.landmark)
            if gesture:
                current_time = time.time()
                if current_time - last_gesture_time > gesture_timeout:
                    finalize_and_speak()  # Speak the sentence after timeout
                recognize_and_speak(gesture)
                last_gesture_time = current_time

                # Display the recognized sentence
                displayed_text = " ".join(recognized_sentence)
                cv2.putText(frame, f'Recognized: {displayed_text}', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Non-Verbal to Verbal Communication', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        finalize_and_speak()  # Ensure last sentence is spoken
        break

cap.release()
cv2.destroyAllWindows()
