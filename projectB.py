import numpy as np
import matplotlib.pyplot as plt 
import cv2 

video_path = 'C:/Users/leahg/OneDrive - Technion/Documents/spring 24/Project B/IMG_1324.mov'

def get_frame_at_time(video_path, target_time):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    # Calculate the target frame number based on the requested time
    target_frame = int(target_time * fps)
    print(f"Target frame: {target_frame}")

    # Set the video capture to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # Read the frame
    ret, frame = cap.read()

    # Release the video capture object
    cap.release()

    # Check if the frame was successfully read
    if not ret:
        print("Failed to read the frame from the video.")
        return None
    else:
        # Convert the frame to RGB format for displaying
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

# Example usage:
video_path = 'C:/Users/leahg/OneDrive - Technion/Documents/spring 24/Project B/IMG_1324.mov'
target_time = 1 * 60 + 38  # Time in seconds

frame_image = get_frame_at_time(video_path, target_time)

def calculate_polygon_area(points):
    # Using the Shoelace formula to calculate the area of the polygon
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def mark_points_and_calculate_area(image, num_points=20):
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.title("Click to mark points")
    plt.axis('off')
    for round in range(2):
        # Use ginput to interactively select points on the image
        points = plt.ginput(num_points, timeout=0)  # timeout=0 means no timeout, waits for user input

        # Draw lines between the selected points
        for i in range(len(points)):
            point1 = points[i]
            point2 = points[(i + 1) % len(points)]  # connect to the next point, and last point connects to the first
            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-')  # 'r-' means red color, solid line
            plt.plot(point1[0], point1[1], 'bo')  # 'bo' means blue color, circle marker for the points
        # Calculate the area of the polygon formed by the selected points
        area = calculate_polygon_area(points)
        print(f"Area of the marked region: {area} square pixels")

    plt.show()
    return points, area

frame_image = get_frame_at_time(video_path, target_time)

# Display the image, mark points, draw lines, and calculate the area if it was successfully retrieved
if frame_image is not None:
    selected_points, area = mark_points_and_calculate_area(frame_image, num_points=20)
    print("Selected points:", selected_points)
    print("Calculated area:", area)