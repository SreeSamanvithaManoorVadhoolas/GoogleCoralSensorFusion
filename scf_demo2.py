# Import libraries
import argparse
import time
import cv2
import sys
from periphery import GPIO
from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# Configure GPIO pins
gpio_out = GPIO("/dev/gpiochip0", 36, "out")  # pin 12
gpio_in = GPIO("/dev/gpiochip0", 9, "in")  # pin 11

def draw_objects(draw, objs, labels, label_list):
    """Draws the bounding box and label for each object. If object is human, box is green, else red"""
    for obj in objs:
        bbox = obj.bbox
        label_string = label_list[obj.id]
        # Drawing of the rectangles of detected objects.
        if obj.id == 0:
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                           outline='green')
            draw.text((bbox.xmin + 10, bbox.ymin + 10),
                      '%s\n%.2f' % (label_string, obj.score),
                      fill='green')
            # Turn on the LED by setting the output pin to True (high)
            gpio_out.write(False)
        else:
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                           outline='red')
            draw.text((bbox.xmin + 10, bbox.ymin + 10),
                      '%s\n%.2f' % (label_string, obj.score),
                      fill='red')

def capture_and_save_image(image_path):
    """This function captures an image by using the Coral Mini Camera"""
    error = False

    # Initialize Google Camera
    cap = cv2.VideoCapture('/dev/video1')

    # Capture a frame
    ret, frame = cap.read()

    if ret:
        # Save the captured frames
        cv2.imwrite(image_path, frame)
    else:
        print("Failed to capture image")
        error = True

    # Release Camera
    cap.release()

    return error

def object_detection():
    """This function runs the object detection TFLite model. First it defines the paths to be used, will then
    take a camera capture and process it into the model."""
    # Definition of the different needed paths
    base_path = "/home/mendel/pycoral/test_data/"
    label_path = base_path + "coco_labels.txt"
    interpreter_path = base_path + "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    capture_path = base_path + "capture.jpg"

    # Upon call of main
    capture_and_save_image(capture_path)
    image_path = capture_path
    output_path = base_path + "capture-processed.jpg"

    # Fetch the label list from coco_labels.txt to a list.
    label_list = []
    with open(label_path, "r") as file:
        # Read each line from the file and append it to the list
        for line in file:
            # Remove newline characters and add the line to the list
            label_list.append(line.strip())

    """Normally the command on line 25 is called with the respective arguments to run a model. In this script
    the model is run inside the same script and therefore, we specify ourselves the parameters. Some we skip
    and are left as placeholders."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=False,
                        help='File path of .tflite file')
    parser.add_argument('-i', '--input', required=False,
                        help='File path of image to process')
    parser.add_argument('-l', '--labels', help='File path of labels file')
    parser.add_argument('-o', '--output',
                        help='File path for the result image with annotations')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects')
    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of times to run inference')
    args = parser.parse_args()

    # Here we fill our arguments manually
    labels = read_label_file(label_path) if args.labels else {}
    interpreter = make_interpreter(interpreter_path)
    interpreter.allocate_tensors()

    # Open camera capture and pre-process it for the model
    image = Image.open(image_path)
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes',
          'loading the model into Edge TPU memory.')

    # Run n rounds (depending on count argument) of the model and print each round's performance.
    for _ in range(args.count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(interpreter, args.threshold, scale)
        print('%.2f ms' % (inference_time * 1000))

    # Print results
    print('-------RESULTS--------')

    # If no objects detected in the picture
    if not objs:
        print('No objects detected')

    # Check if a human is detected
    human_present = False

    # Print the results of the detected objects and the score. It also prints the box coordinates.
    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  object:    ', label_list[obj.id])
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

        # Check if the detected object is a human
        if obj.id == 0:
            human_present = True

    # Draw the rectangles over the capture and save it into the output path
    image = image.convert('RGB')
    draw_objects(ImageDraw.Draw(image), objs, labels, label_list)
    image.save(output_path)
    # image.show()

    return human_present


# Main
if __name__ == '__main__':
    """This main function is in charge of detecting the light presence and triggering the object detection model"""
    try:
        var = True

        while True:
            # Check if human is detected
            human_detected = object_detection()

            if human_detected:
                print("Human Detected")
                print("Checking Sensor Data....")

                # Check if Carbon Monoxide level is high
                while True:
                    input_value = gpio_in.read()
                    if var:
                        if input_value:
                            # If.
                            gpio_out.write(True)
                            print("Fire Detected!!")
                            # object_detection()
                            var = False
                            time.sleep(5)

                    else:
                        # Turn off the LED by setting the output pin to False (low)
                        time.sleep(5)
                        gpio_out.write(False)
                        sys.exit()

            # Delay for a short period of time
            time.sleep(0.1)

    except KeyboardInterrupt:
        # Clean up GPIO pins and exit
        gpio_in.close()
        gpio_out.close()
