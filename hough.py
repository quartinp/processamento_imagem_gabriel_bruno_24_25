import cv2
import numpy as np
import matplotlib.pyplot as plt


##### IMAGENS USADAS
img_cal = "C:/Users/gabri/OneDrive/Documentos/Universidade/2o semestre/SEIM/imgs/img2.jpeg"
img_test = "C:/Users/gabri/OneDrive/Documentos/Universidade/2o semestre/SEIM/imgs/img1.jpeg"


## PARÂMETROS PARA A FUNC HOUGHLINESP
img_test = cv2.imread(img_test)
img_cal = cv2.imread(img_cal)

gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
gray_cal = cv2.cvtColor(img_cal, cv2.COLOR_BGR2GRAY)

    #blurred = cv2.GaussianBlur(gray, (5,5), 0)

edges = cv2.Canny(gray, 50, 150)
edges_cal = cv2.Canny(gray_cal, 50, 150)

thresh = 10 
min_comprimento=10
gap_max=50


## Calibração
def calibrate(image_path):

    global img_cal, gray, edges, thresh, min_comprimento, gap_max
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = thresh, minLineLength = min_comprimento, maxLineGap = gap_max)
    print(lines)

    if lines is None:
        raise ValueError("No lines detected. Adjust Hough parameters or check calibration image.")

    horizontal = []
    vertical = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 5:
            horizontal.append((min(x1, x2), max(x1, x2), y1))
        elif abs(x1 - x2) < 7:
            vertical.append((min(y1, y2), max(y1, y2), x1))

    print("Horizontais (calibracao): ", horizontal)
    print ("Verticais (calibracao): ", vertical)
    
    if len(horizontal) >= 2 and len(vertical) >= 2:
        horizontal_sorted = sorted(horizontal, key=lambda x: x[0])
        vertical_sorted = sorted(vertical, key=lambda x: x[2])
        width_px = abs(horizontal_sorted[0][0] - horizontal_sorted[0][1])
        height_px = abs(vertical_sorted[0][0] - vertical_sorted[0][1])
        mm_per_pixel = 50.0 / max(width_px, height_px)  # Assuming 50mm calibration square
        return mm_per_pixel
    else:
        raise ValueError("Calibration failed. Check calibration image.")

def detect_objects(image_path, mm_per_pixel):

    global img_test, gray, blurred, edges

    plt.imshow(edges)
    plt.axis('off')
    plt.show()

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = thresh, minLineLength = min_comprimento, maxLineGap = gap_max)
    if lines is None:
        return []

    horizontal_lines = []
    vertical_lines = []


    ## ISTO AQUI SEPARA EM HORIZONTAIS E VERTICAIS, VAI TER DE SER ADAPTADO QUANDO FOR LIDAR COM LINHAS NÃO PARALELAS
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 5:  # Horizontal line
            x_start, x_end = sorted([x1, x2])
            horizontal_lines.append((x_start, x_end, int((y1-y2)/2)))   # escolhi o y médio dos dois pontos para o y comum
        elif abs(x1 - x2) < 5:  # Vertical line
            y_start, y_end = sorted([y1, y2])
            vertical_lines.append(    (y_start, y_end, int((x1-x2)/2))    )
    
    rectangles = []
    tolerance = 10  # Allowed pixel difference for alignment
    
    for i in range(len(horizontal_lines)):
        h_top = horizontal_lines[i]
        for j in range(i+1, len(horizontal_lines)):
            h_bottom = horizontal_lines[j]
            if h_top[2] > h_bottom[2]:
                h_top, h_bottom = h_bottom, h_top
            if (abs(h_top[0] - h_bottom[0]) < tolerance) and (abs(h_top[1] - h_bottom[1]) < tolerance):
                left_x = h_top[0]
                right_x = h_top[1]
                top_y = h_top[2]
                bottom_y = h_bottom[2]
                
                left_v = None
                right_v = None
                for v in vertical_lines:
                    v_y1, v_y2, v_x = v
                    if abs(v_x - left_x) < tolerance:
                        if (v_y1 <= top_y + tolerance) and (v_y2 >= bottom_y - tolerance):
                            left_v = v
                    if abs(v_x - right_x) < tolerance:
                        if (v_y1 <= top_y + tolerance) and (v_y2 >= bottom_y - tolerance):
                            right_v = v
                if left_v and right_v:
                    width = (right_x - left_x) * mm_per_pixel
                    height = (bottom_y - top_y) * mm_per_pixel
                    x_center = (left_x + right_x)/2 * mm_per_pixel
                    y_center = (top_y + bottom_y)/2 * mm_per_pixel
                    perimeter = 2 * (width + height)
                    area = width * height
                    
                    rectangles.append({
                        'center': (x_center, y_center),
                        'perimeter': perimeter,
                        'area': area,
                        'width': width,
                        'height': height,
                        'std_dev': 0.0,
                        'pixel_coords': {  # Added for visualization
                            'top_left': (left_x, top_y),
                            'bottom_right': (right_x, bottom_y)
                        }
                    })
    return rectangles

# Visualization function
def draw_rectangles(image_path, rectangles):
    global img_test

    for rect in rectangles:
        pt1 = tuple(int(x) for x in rect['pixel_coords']['top_left'])
        pt2 = tuple(int(x) for x in rect['pixel_coords']['bottom_right'])
        cv2.rectangle(img_test, pt1, pt2, (0, 255, 0), 2)  # Green rectangle with 2px thickness
        
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def draw_all_lines():

    global img_test
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = thresh, minLineLength = min_comprimento, maxLineGap = gap_max)
    
    if lines is not None:
        # Classify and draw lines with different colors
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Classify and color code
            if abs(angle) < 5:  # Horizontal (red)
                cv2.line(img_test, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif abs(angle - 90) < 5:  # Vertical (blue)
                cv2.line(img_test, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:  # Other angles (yellow)
                cv2.line(img_test, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # Create figure with annotations
    plt.figure(figsize=(12, 12))
    plt.title("Detected Lines\n(Red=Horizontal, Blue=Vertical, Yellow=Other)", fontsize=14, pad=20)
    plt.imshow(img_test)
    plt.axis('off')
    
    # Add statistics text box
    text_str = f"Total lines: {len(lines)}\n"
    text_str += f"Horizontal: {len([l for l in lines if abs(np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]))) < 5])}\n"
    text_str += f"Vertical: {len([l for l in lines if abs(np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) - 90) < 5])}"
    
    plt.gcf().text(0.1, 0.95, text_str, 
                  fontsize=12, 
                  bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()


# Example usage
mm_per_pixel = calibrate(img_cal)
results = detect_objects(img_test, mm_per_pixel)
draw_rectangles(img_test, results)
draw_all_lines()

print(results)
