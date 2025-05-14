# Importing Required Libraries
import cv2 as opencv
import mediapipe as mediapipe
import subprocess as macosScripts

VOLUME_STEP = 7 # How much volume will be change each time!
GESTURE_THRESHOLD = 0.05 # Sensitivity of gesture detection!

# Function - For Controlling/Setting the macOS Volume Up/Down
def change_macos_volume(direction):
    """
    Changes macOS Volume Up/Down Using Apple Scripts.
    In parameters it receives direction: up  or down 
    """

    # Here in below lines of code: 'output volume of (get volume settings)' 
    # give us the current volume of the system
    if direction == 'up':
        script = f"set volume output volume (output volume of (get volume settings) + {VOLUME_STEP})"
    elif direction == 'down':
        script = f"set volume output volume (output volume of (get volume settings) - {VOLUME_STEP})"
    else:
        print(f"Invalid Volume Direction: {direction}")
        return
    
    # Here in below lines of code: Executing the scripts of macOS
    try:
        macosScripts.run(['osascript','-e',script], check=True, text=True, capture_output=True)
        # check is for: will raise the CalledProcessError, If osascript returns the non-zero exit code.
        # text & capture_output is for: Debugging Purpose.
    except macosScripts.CalledProcessError:
        # Error Occur when volume up/down reaches the max/min.
        pass
    except FileNotFoundError:
        print("Error: 'osascript' command not found.")


# Initialzing the mediapipe hands tool
mediapipe_hands_solution = mediapipe.solutions.hands
hands_detector = mediapipe_hands_solution.Hands(
    static_image_mode = False, # Not an image: it will be a video.
    max_num_hands = 1, # Detect only one hand.
    min_detection_confidence = 0.6, 
    min_tracking_confidence = 0.6
)
mediapipe_drawing_utils = mediapipe.solutions.drawing_utils # Utility for drawing the hand landmarks

# Opening the webcam for capturing video using opencv
video_capture = opencv.VideoCapture(0)

if not video_capture.isOpened():
    print('Video is not captured: Please try again!')
    exit()

# Loop will be running until and unless the video is capturing
while video_capture.isOpened():

    success, frame = video_capture.read() # Success will store 0/1 or True/False: When video is capture or not
    # frame contain the video frame

    if not success:
        print('Ignoring the empty frame')
        continue

    # Flipping the frame: Its for showing the exact user hand in video.
    # If user is using right hand than using it right hand will be shown otherwise lefthand will be shown in video
    frame = opencv.flip(frame, 1)

    # Converting the image into RGB (Mediapipe is using RGB format not BGR)
    rgb_frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)
    

    # Processing the RGB Image using mediapipe
    detection_result = hands_detector.process(rgb_frame)

    # Checking if hands marks were detected or not
    if detection_result.multi_hand_landmarks:
        print('hand landmarks detected!')

        # Iterating the loop on each hand landmark
        for hand_landmarks in detection_result.multi_hand_landmarks:
            
            # Drawing the landmarks & connnection on original BGR Frame
            mediapipe_drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mediapipe_hands_solution.HAND_CONNECTIONS # Draw lines conneting the landmarks
            )

            # Give us the values of the index_finder_tip & thumb_tip 'these values are like: top-bottom from 0.0-1.0'
            # For Seeing it visually: check out the mediapipe docs in the hand landmarks tool & search about
            # its visual image.
            index_finger_tip_y = hand_landmarks.landmark[mediapipe_hands_solution.HandLandmark.INDEX_FINGER_TIP].y
            thumb_tip_y = hand_landmarks.landmark[mediapipe_hands_solution.HandLandmark.THUMB_TIP].y

            # This line calculates how much higher or lower the index finger is compared to the thumb. 
            vertical_diff = index_finger_tip_y - thumb_tip_y

            # SUMMARY
            # In MediaPipe, smaller y values mean higher on the screen.
            # We subtract thumb_y from index_finger_y to see the vertical difference.
            # If the index finger is higher (negative diff), we increase volume.
            # If the index finger is lower (positive diff), we decrease volume.

            if vertical_diff < -GESTURE_THRESHOLD:
                print('Gesture: Pointing Up')
                change_macos_volume('up')
            elif vertical_diff > GESTURE_THRESHOLD:
                print('Gesture: Pointing Down')
                change_macos_volume('down')

             # Displaying the frame 
            opencv.imshow('Hand Gesture Volume Control', frame)

            # Quitting if user click on 'q' and wait for the 5 milliseconds
            if opencv.waitKey(5) & 0xFF == ord('q'):
                print('Quitting')
                break


    else:
        print('No hand landmarks detected!')
    

video_capture.release() # Releasing the video capture
opencv.destroyAllWindows() # Destroying all the open windows
hands_detector.close() # Releasing mediapipe hands resources


