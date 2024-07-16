import argparse
from palm_lines_recognition.tools import *
from palm_lines_recognition.model import *
from palm_lines_recognition.rectification import *
from palm_lines_recognition.detection import *
from palm_lines_recognition.classification import *
from palm_lines_recognition.measurement import *


def run(path_to_input_image, resize_value, output_folder):
    path_to_clean_image = '{}/palm_without_background.jpg'.format(output_folder)
    path_to_warped_image = '{}/warped_palm.jpg'.format(output_folder)
    path_to_warped_image_clean = '{}/warped_palm_clean.jpg'.format(output_folder)
    path_to_warped_image_mini = '{}/warped_palm_mini.jpg'.format(output_folder)
    path_to_warped_image_clean_mini = '{}/warped_palm_clean_mini.jpg'.format(output_folder)
    path_to_palmline_image = '{}/palm_lines.png'.format(output_folder)
    path_to_result = '{}/result.jpg'.format(output_folder)

    # 0. Preprocess image
    remove_background(path_to_input_image, path_to_clean_image)

    # 1. Palm image rectification
    warp_result = warp(path_to_input_image, path_to_warped_image)
    if warp_result is None:
        print_error()
    else:
        remove_background(path_to_warped_image, path_to_warped_image_clean)
        resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value)

        # 2. Principal line detection
        detect(path_to_warped_image_clean, path_to_palmline_image, resize_value)

        # 3. Line classification
        lines = classify(path_to_palmline_image)

        # 4. Length measurement
        im, heart_line_length, head_line_length, life_line_length, line_descriptions = measure(path_to_warped_image_mini, lines)

        # 5. Save result
        im.save(path_to_result)

        return path_to_result, heart_line_length, head_line_length, life_line_length, line_descriptions
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='the path to the input')
    args = parser.parse_args()
    run(args.input)