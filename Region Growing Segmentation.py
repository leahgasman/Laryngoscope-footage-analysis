import cv2
import itertools
import numpy as np
import random

# Stack class definition
class Stack():
    def __init__(self):
        self.item = []

    def push(self, value):
        self.item.append(value)

    def pop(self):
        return self.item.pop()

    def size(self):
        return len(self.item)

    def isEmpty(self):
        return self.size() == 0

    def clear(self):
        self.item = []


# regionGrow class definition
class regionGrow():
    def __init__(self, frame, th):
        self.im = frame.astype('int')  # Use the frame directly
        self.h, self.w, _ = self.im.shape
        self.passedBy = np.zeros((self.h, self.w), np.double)
        self.currentRegion = 0
        self.iterations = 0
        self.SEGS = np.zeros((self.h, self.w, 3), dtype='uint8')
        self.stack = Stack()
        self.thresh = float(th)

    def getNeighbour(self, x0, y0):
        neighbours = []
        for i, j in itertools.product((-1, 0, 1), repeat=2):
            if (i, j) != (0, 0):
                x = x0 + i
                y = y0 + j
                if self.boundaries(x, y):
                    neighbours.append((x, y))
        return neighbours

    def create_seeds(self, num_seeds=10):
        seeds = []
        for _ in range(num_seeds):
            x = random.randint(0, self.h - 1)
            y = random.randint(0, self.w - 1)
            seeds.append([x, y])
        return seeds

    def ApplyRegionGrow(self, cv_display=True):
        randomseeds = self.create_seeds()  # Random seeds instead of predefined ones
        np.random.shuffle(randomseeds)

        for x0 in range(self.h):
            for y0 in range(self.w):
                if self.passedBy[x0, y0] == 0:
                    self.currentRegion += 1
                    self.passedBy[x0, y0] = self.currentRegion
                    self.stack.push((x0, y0))
                    self.prev_region_count = 0

                    while not self.stack.isEmpty():
                        x, y = self.stack.pop()
                        self.BFS(x, y)
                        self.iterations += 1

                        # Debug: Display segmentation progress every 1000 iterations
                        if self.iterations % 1000 == 0:  # Show progress every 1000 iterations
                            print(f"Iterations: {self.iterations}, Region: {self.currentRegion}")
                            cv2.imshow("Region Growing Progress", self.SEGS)
                            cv2.waitKey(1)  # Allows for real-time display updates

                    if self.PassedAll():
                        break

                    if self.prev_region_count < 8 * 8:
                        x0, y0 = self.reset_region(x0, y0)

        if self.iterations > 20000:
            print("Max Iterations reached (20,000).")
        print(f"Iterations : {str(self.iterations)}")

        if cv_display:
            [self.color_pixel(i, j) for i, j in itertools.product(range(self.h), range(self.w))]
            self.display()

    def reset_region(self, x0, y0):
        self.passedBy[self.passedBy == self.currentRegion] = 0
        x0 = random.randint(x0 - 4, x0 + 4)
        y0 = random.randint(x0 - 4, x0 + 4)
        x0 = np.clip(x0, 0, self.h - 1)
        y0 = np.clip(y0, 0, self.w - 1)
        self.currentRegion -= 1
        return x0, y0

    def color_pixel(self, i, j):
        val = self.passedBy[i][j]
        # Check if the pixel is part of a valid region
        if val != 0:
            self.SEGS[i][j] = (255, 255, 255)  # White for valid regions
        else:
            self.SEGS[i][j] = (0, 0, 0)  # Black for background

    def display(self):
        # Display the segmentation result
        cv2.imshow("Segmentation Result", self.SEGS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def BFS(self, x0, y0):
        regionNum = self.passedBy[x0, y0]
        elems = [np.mean(self.im[x0, y0])]
        var = self.thresh
        neighbours = self.getNeighbour(x0, y0)

        for x, y in neighbours:
            if self.passedBy[x, y] == 0 and self.distance(x, y, x0, y0) < var:
                if self.PassedAll():
                    break
                self.passedBy[x, y] = regionNum
                self.stack.push((x, y))
                elems.append(np.mean(self.im[x, y]))
                var = np.var(elems)
                self.prev_region_count += 1
            var = max(var, self.thresh)

    def PassedAll(self, max_iteration=20000):
        return self.iterations >= max_iteration or np.all(self.passedBy > 0)

    def boundaries(self, x, y):
        return 0 <= x < self.h and 0 <= y < self.w

    def distance(self, x, y, x0, y0):
        # Convert pixels to grayscale values for distance calculation
        pixel1 = self.im[x0, y0]
        pixel2 = self.im[x, y]
        # Convert to grayscale by averaging the RGB channels
        gray1 = np.mean(pixel1)
        gray2 = np.mean(pixel2)
        return abs(gray1 - gray2)  # Use absolute difference for thresholding


# Function to extract a frame from a video
def get_frame_at_time(video_path, target_time):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(target_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read the frame from the video.")
        return None
    else:
        return frame


# Example usage
video_path = r'C:\Users\avivd\Desktop\EE\Semester H\Project_B\videos\first_vid.mov'
frame = get_frame_at_time(video_path, target_time=89)  # Get the frame at 2 seconds

if frame is not None:
    # Display the extracted frame to ensure it's loaded correctly
    cv2.imshow("Extracted Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply region growing
    print("arrive to region growing")
    region_grower = regionGrow(frame, th=30)
    region_grower.ApplyRegionGrow(cv_display=True)

    # Check final segmentation
    cv2.imshow("Final Segmentation", region_grower.SEGS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to extract the frame.")
