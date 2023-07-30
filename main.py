
import cv2
import face_recognition_utils as fr_utils
import video_capture_utils as vc_utils

# Load reference images
reference_images = {
    'Tony Soprano': 'images/tony-soprano.jpg',
    'Robert De Niro': 'images/robert-de-niro.jpeg',
    'Marlo Stanfield': 'images/marlo-stanfield.jpeg',
    'Vito Spatafore': 'images/vito-spatafore.jpeg',
}

# Load reference face encodings
reference_face_encodings = fr_utils.load_images(reference_images)

# Initialize video capture
video_capture = vc_utils.start_video_capture()

# Process video frames for face recognition
while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    recognized_faces = fr_utils.recognize_faces(frame, reference_face_encodings)

    # Draw rectangles and labels on the frame
    for (top, right, bottom, left), name in recognized_faces:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame with recognized faces
    cv2.imshow('Face Recognition', frame)

    # Exit the loop on a key press (e.g., 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
vc_utils.release_video_capture(video_capture)
