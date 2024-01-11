import cv2
import numpy as np



def apply_horizontal_dilation(image, kernel_length=500):
    # Create a horizontal kernel
    kernel = np.ones((1, kernel_length), np.uint8)

    # Apply dilation
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    return dilated_image

def apply_sobel_gradient_with_intensity_and_angle(frame):
    """Apply Sobel gradient with intensity and angle calculations to a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # Normalize magnitude for display
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    return magnitude, angle

def custom_filter(frame):
    """Apply custom filter based on the condition g > r > b."""
    b, g, r = cv2.split(frame)
    condition = np.logical_and(g > r, r > b)
    binary_image = np.where(condition, 0, 1).astype(np.uint8) * 255
    return binary_image


def detect_lines(processed_frame, frame_shape):
    """Create an image with detected lines."""
    # Create a blank image to draw lines on
    line_image = np.zeros_like(processed_frame)

    # Parameters for HoughLinesP
    rho = 1  # Distance resolution in pixels
    theta = np.pi / 180  # Angular resolution in radians
    threshold = 150  # Threshold for line detection
    minLineLength = 50  # Minimum length of line
    maxLineGap = 10  # Maximum allowed gap between line segments

    edges = cv2.Canny(processed_frame, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(dilated, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

    cv2.imshow("Lines",line_image)
    return line_image


def label_frame(original_frame, processed_frame):
    """Label the original frame based on the regions identified in the processed frame."""
    # Find contours in the processed frame
    cv2.imshow("Processed frame",processed_frame)
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    labeled_frame = original_frame.copy()
    for i, contour in enumerate(contours):
        # You can also use boundingRect or minAreaRect to draw rectangles or rotated rectangles
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(labeled_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(labeled_frame, f"Object {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return labeled_frame


def display_combined_frame(sobel_frame, custom_frame):
    """Display combined result of Sobel filtered and custom filtered frames."""
    # Ensure both frames are of the same data type
    sobel_frame = sobel_frame.astype(np.uint8)
    custom_frame = custom_frame.astype(np.uint8)

    # Add the two frames
    combined_frame = cv2.add(sobel_frame, custom_frame)

    return combined_frame








def process_video(video_path):
    """Process a video file with Sobel and custom filter, then combine and display results."""
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        sobel_intensity, _ = apply_sobel_gradient_with_intensity_and_angle(frame)
        custom_filtered = custom_filter(frame)
        
        # Display combined frame
        combined_frame = display_combined_frame(sobel_intensity, custom_filtered)
        line_image = detect_lines(combined_frame, frame.shape)

        # Subtract the line image from the combined frame
        subtracted_frame = cv2.subtract(combined_frame, line_image)


        #subtracted_frame = cv2.subtract(subtracted_frame,averageCustomed)

        cv2.imshow("subtracted_frame",subtracted_frame)


        subtracted_frame = cv2.bitwise_not(subtracted_frame)

        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply morphological opening
        subtracted_frame = cv2.morphologyEx(subtracted_frame, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(subtracted_frame.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area_threshold = 10000  # Set your desired maximum area threshold
        min_area_threshold = 200  # Set your desired minimum area threshold


        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)

            # Check if the contour area is smaller than the threshold
            if area  > min_area_threshold and area < max_area_threshold:
                # Draw the contour if it's smaller than the threshold
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

                # Place a label (optional)
                x, y, w, h = cv2.boundingRect(contour)
                # You can draw the label here if needed


        # Show the original frame with contours and labels
        cv2.imshow('Processed Frame', frame)
        
        # Wait for a key press to move to the next frame or exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Replace 'path_to_video.mp4' with the path to your video file
process_video('1.mp4')

