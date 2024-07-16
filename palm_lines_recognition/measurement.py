from PIL import Image, ImageDraw
import cv2
import mediapipe as mp
import math


# Function to calculate Euclidean distance between two points
def distance_between_points(p1, p2):
    return round(math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))


def measure(path_to_warped_image_mini, lines):
    heart_thres_x = 0
    head_thres_x = 0
    life_thres_y = 0

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(path_to_warped_image_mini), 1)
        image_height, image_width, _ = image.shape

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        hand_landmarks = results.multi_hand_landmarks[0]

        zero = hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y
        one = hand_landmarks.landmark[mp_hands.HandLandmark(1).value].y
        five = hand_landmarks.landmark[mp_hands.HandLandmark(5).value].x
        nine = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].x
        thirteen = hand_landmarks.landmark[mp_hands.HandLandmark(13).value].x

        heart_thres_x = image_width * (1 - (nine + (five - nine) * 2 / 5))
        head_thres_x = image_width * (1 - (thirteen + (nine - thirteen) / 3))
        life_thres_y = image_height * (one + (zero - one) / 3)

    im = Image.open(path_to_warped_image_mini)
    width = 3
    if (None in lines) or (len(lines) < 3):
        return None, None
    else:
        draw = ImageDraw.Draw(im)

        heart_line = lines[0]
        head_line = lines[1]
        life_line = lines[2]

        heart_line_points = [tuple(reversed(l[:2])) for l in heart_line]
        heart_line_tip = heart_line_points[0]
        if heart_line_tip[0] < heart_thres_x:
            heart_content = 'Your Heart line is long, which means you will have long partnership with whom you love or care.'
        else:
            heart_content = 'Your Heart line is short, which means you will meet various people and have a broad range of relationships throughout your life.'
        draw.line(heart_line_points, fill="red", width=width)

        head_line_points = [tuple(reversed(l[:2])) for l in head_line]
        head_line_tip = head_line_points[-1]
        if head_line_tip[0] > head_thres_x:
            head_content = 'Your Head line is long, which means you will explore a broad range of topics throughout your life.'
        else:
            head_content = 'Your Head line is short, which means you will be fascinated by one topic and dig deep into it.'
        draw.line(head_line_points, fill="green", width=width)

        life_line_points = [tuple(reversed(l[:2])) for l in life_line]
        life_line_tip = life_line_points[-1]
        if life_line_tip[1] > life_thres_y:
            life_content = 'Your Life line is long, which means you tend to solve problems with other people rather than by yourself.'
        else:
            life_content = 'Your Life line is short, which means you are independent and autonomous.'
        draw.line(life_line_points, fill="blue", width=width)

        heart_line_length = distance_between_points(heart_line_points[0], heart_line_points[-1])
        head_line_length = distance_between_points(head_line_points[0], head_line_points[-1])
        life_line_length = distance_between_points(life_line_points[0], life_line_points[-1])

        # draw.line([(heart_thres_x, 0), (heart_thres_x, image_height)], fill="red")
        # draw.line([(head_thres_x, 0), (head_thres_x, image_height)], fill="green")
        # draw.line([(0, life_thres_y), (image_width, life_thres_y)], fill="blue")
        line_descriptions = {'heart_content': heart_content, 'head_content': head_content, 'life_content': life_content}
        return im, heart_line_length, head_line_length, life_line_length, line_descriptions