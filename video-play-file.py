import cv2

# Create a VideoCapture object
# cap = cv2.VideoCapture('record/mjpeg-4k.mp4')
cap = cv2.VideoCapture('record/h265-4k.mp4')

# Create a named window with a specific size
cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Window', 800, 600)  # Set the window size to 800x600

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow('Window', frame)

    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()