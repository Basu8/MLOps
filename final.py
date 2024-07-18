# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from collections import deque
# import gradio as gr
# from PIL import Image
# from keras.models import load_model

# # Constants
# IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
# SEQUENCE_LENGTH = 20
# CLASSES_LIST_VIDEO = ['HorseRace', 'VolleyballSpiking', 'Biking', 'TaiChi', 'Punch', 'BreastStroke', 'Billiards', 'PoleVault', 'ThrowDiscus', 'BaseballPitch', 'HorseRiding', 'Mixing', 'HighJump', 'Skijet', 'SkateBoarding', 'MilitaryParade', 'Fencing', 'JugglingBalls', 'Swing', 'RockClimbingIndoor', 'SalsaSpin', 'PlayingTabla', 'Rowing', 'BenchPress', 'PushUps', 'Nunchucks', 'PlayingViolin']
# CLASSES_LIST_IMAGE = ['Sitting', 'Using laptop', 'Hugging', 'Sleeping', 'Drinking', 'Clapping', 'Dancing', 'Cycling', 'Calling', 'Laughing', 'Eating', 'Fighting', 'Listening to music', 'Running', 'Texting']

# # Load the trained models
# video_model_path = 'model_final.h5'  # Replace with your actual video model path
# image_model_path = 'efficientnet_model.keras'  # Replace with your actual image model path
# LRCN_model = tf.keras.models.load_model(video_model_path)
# image_model = load_model(image_model_path)

# # Emoji mapping for image prediction
# class_to_emoji = {
#     'Sitting': '😌',
#     'Using laptop': '💻',
#     'Hugging': '🤗',
#     'Sleeping': '😴',
#     'Drinking': '🍹',
#     'Clapping': '👏',
#     'Dancing': '💃',
#     'Cycling': '🚴‍♂️',
#     'Calling': '📱',
#     'Laughing': '😂',
#     'Eating': '🍽️',
#     'Fighting': '👊',
#     'Listening to music': '🎧',
#     'Running': '🏃‍♂️',
#     'Texting': '✉️'
# }

# def frames_extraction(video_path):
#     frames_list = []
#     video_reader = cv2.VideoCapture(video_path)
#     video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
#     skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

#     for frame_counter in range(SEQUENCE_LENGTH):
#         video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
#         success, frame = video_reader.read()
#         if not success:
#             break
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
#         normalized_frame = resized_frame / 255
#         frames_list.append(normalized_frame)

#     video_reader.release()
#     return frames_list

# def predict_on_video(video_file_path):
#     output_file_path = video_file_path + '_output.mp4'
#     video_reader = cv2.VideoCapture(video_file_path)
#     original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
#                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
#     frames_queue = deque(maxlen=SEQUENCE_LENGTH)
#     predicted_class_name = ''
    
#     while video_reader.isOpened():
#         ok, frame = video_reader.read()
#         if not ok:
#             break
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
#         normalized_frame = resized_frame / 255
#         frames_queue.append(normalized_frame)
#         if len(frames_queue) == SEQUENCE_LENGTH:
#             predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
#             predicted_label = np.argmax(predicted_labels_probabilities)
#             predicted_class_name = CLASSES_LIST_VIDEO[predicted_label]
#         cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         video_writer.write(frame)
    
#     video_reader.release()
#     video_writer.release()
#     return predicted_class_name

# def predict_on_image(image):
#     img = Image.fromarray(image.astype('uint8'), 'RGB')  # Convert NumPy array to PIL Image
#     img = img.resize((160, 160))  # Resize the image to match model input size
#     img_array = np.asarray(img) / 255.0  # Normalize the image
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     result = image_model.predict(img_array)
#     itemindex = np.argmax(result)
#     prediction = CLASSES_LIST_IMAGE[itemindex]
#     emoji = class_to_emoji.get(prediction, '❓')  # Get emoji corresponding to predicted class, default to question mark if not found
    
#     return emoji

# def predict(choice, video=None, image=None):
#     if choice == "Video":
#         if video is None:
#             return "Please upload a video."
#         predicted_class_name = predict_on_video(video)  # Use video path directly
#         return predicted_class_name
#     elif choice == "Image":
#         if image is None:
#             return "Please upload an image."
#         emoji = predict_on_image(image)
#         return emoji
#     else:
#         return "Invalid choice."

# with gr.Blocks() as demo:
#     with gr.Row():
#         radio = gr.Radio(choices=["Video", "Image"], label="Select Input Type")
    
#     with gr.Row():
#         input_video = gr.Video(label="Input Video", visible=False)
#         input_image = gr.Image(label="Input Image", visible=False)
        
#         radio.change(
#             fn=lambda x: (gr.update(visible=x == "Video"), gr.update(visible=x == "Image")),
#             inputs=radio,
#             outputs=[input_video, input_image]
#         )
    
#     with gr.Row():
#         output_text = gr.Textbox(label="Output Text", visible=True)
    
#     predict_button = gr.Button("Predict")
    
#     predict_button.click(
#         fn=predict,
#         inputs=[radio, input_video, input_image],
#         outputs=[output_text]
#     )

# demo.launch()




# import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import gradio as gr
from PIL import Image
from keras.models import load_model

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST_VIDEO = ['HorseRace', 'VolleyballSpiking', 'Biking', 'TaiChi', 'Punch', 'BreastStroke', 'Billiards', 'PoleVault', 'ThrowDiscus', 'BaseballPitch', 'HorseRiding', 'Mixing', 'HighJump', 'Skijet', 'SkateBoarding', 'MilitaryParade', 'Fencing', 'JugglingBalls', 'Swing', 'RockClimbingIndoor', 'SalsaSpin', 'PlayingTabla', 'Rowing', 'BenchPress', 'PushUps', 'Nunchucks', 'PlayingViolin']
CLASSES_LIST_IMAGE = ['Sitting', 'Using laptop', 'Hugging', 'Sleeping', 'Drinking', 'Clapping', 'Dancing', 'Cycling', 'Calling', 'Laughing', 'Eating', 'Fighting', 'Listening to music', 'Running', 'Texting']

# Load the trained models
video_model_path = 'model_final.h5'  # Replace with your actual video model path
image_model_path = 'efficientnet_model.h5'  # Replace with your actual image model path
LRCN_model = tf.keras.models.load_model(video_model_path)
efficientnet_model = load_model(image_model_path)

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

def predict_on_video(video_file_path):
    output_file_path = video_file_path + '_output.mp4'
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST_VIDEO[predicted_label]
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        video_writer.write(frame)
    
    video_reader.release()
    video_writer.release()
    return predicted_class_name

def read_img(fn):
    # img = Image.open(fn)
    return np.asarray(fn.resize((160,160)))

def predict_on_image(test_image):
    try:
        # If test_image is a NumPy array, convert it to a PIL Image
        if isinstance(test_image, np.ndarray):
            test_image = Image.fromarray(test_image)

        # If test_image is not a PIL Image at this point, raise an error
        if not isinstance(test_image, Image.Image):
            raise ValueError("Input is not a PIL Image or convertible to a PIL Image")

        # Resize the image to match model input size
        resized_image = test_image.resize((160, 160))

        # Convert the resized image to a NumPy array
        image_array = np.asarray(resized_image, dtype='uint8')

        # Ensure the image array has 3 channels (RGB)
        if image_array.shape[-1] != 3:
            raise ValueError("Input image does not have 3 channels (RGB)")

        # Expand dimensions to match the model input shape
        image_array_expanded = np.expand_dims(image_array, axis=0)

        # Make prediction using the model
        result = efficientnet_model.predict(image_array_expanded)

        # Get the index of the highest probability class
        prediction = np.argmax(result)

        # Get the probability of the predicted class
        probability = np.max(result) * 100

        # Get the predicted class name from the class list
        predicted_class = CLASSES_LIST_IMAGE[prediction]

        return predicted_class
    except Exception as e:
        print(f"Error: {e}")
        return None, None
def predict(choice, video=None, image=None):
    if choice == "Video":
        if video is None:
            return "Please upload a video."
        predicted_class_name = predict_on_video(video)  # Use video path directly
        return predicted_class_name
    elif choice == "Image":
        if image is None:
            return "Please upload an image."
        activity = predict_on_image(image)
        return activity
    else:
        return "Invalid choice."

with gr.Blocks() as demo:
    gr.Markdown("""
        <div style="text-align: center;">
            <h1> Image and Video Classification </h1>
            Upload an image or video to get the predicted class using EfficientNet for images and LRCN for videos.
        </div>
    """)    
    
    with gr.Row():
        radio = gr.Radio(choices=["Video", "Image"], label="Select Input Type")
    
    with gr.Row():
        input_video = gr.Video(label="Input Video", visible=False)
        input_image = gr.Image(label="Input Image", visible=False)
        
        radio.change(
            fn=lambda x: (gr.update(visible=x == "Video"), gr.update(visible=x == "Image")),
            inputs=radio,
            outputs=[input_video, input_image]
        )
    
    with gr.Row():
        output_text = gr.Textbox(label="Output Text", visible=True)
    
    predict_button = gr.Button("Predict")
    
    predict_button.click(
        fn=predict,
        inputs=[radio, input_video, input_image],
        outputs=[output_text]
    )
    
demo.launch()

