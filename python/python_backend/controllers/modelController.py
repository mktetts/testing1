from python_backend.utils.success import Success
from python_backend.utils.error import Error
import python_backend.contract.blockchain
from python_backend.contract.admin import doctorDetailsInstance
from python_backend.contract.deploy import deploy_contract as deploy
import python_backend.contract.deploy
from web3.middleware import geth_poa_middleware
from transformers import TapasTokenizer
import scipy

from flask import jsonify, request
import sys, os, pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
import tensorflow as tf
from PIL import Image
import cv2, io, librosa
from io import BytesIO
import openvino as ov

sys.dont_write_bytecode = True

def predictWound():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['image']

    core = Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))

    model_xml = os.path.join(current_directory, '..', 'trained_models', 'wound', 'optimized_wound.xml')
    # model_xml = os.path.join(os.path.dirname(__file__), 'DoctorDetailContract.json')
    quantized_model = core.read_model(model_xml)

    quantized_compiled_model = core.compile_model(model=quantized_model, device_name="CPU")
    def pre_process_image(imagePath, img_height=180):
        n, c, h, w = [1, 3, img_height, img_height]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)

        image = np.array(image)

        input_image = image.reshape((n, h, w, c))

        return input_image

    input_layer = quantized_compiled_model.input(0)
    output_layer = quantized_compiled_model.output(0)

    class_file = os.path.join(current_directory, '..', 'trained_models', 'wound', 'wound_class_list.pkl')

    with open(class_file, 'rb') as file:
        loaded_list = pickle.load(file)
    class_names = loaded_list
   
    input_image = pre_process_image(imagePath=image_file)

    res = quantized_compiled_model([input_image])[output_layer]

    score = tf.nn.softmax(res[0])

    prediction = "This wound is likely belongs to {} with a {:.2f} percent confidence.".format(
            class_names[np.argmax(score)], 100 * np.max(score)
        )
    return Success("Success", prediction, 200)

def predictMedicine(image_file):
    
    # image_file = request.files['image']

    core = Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))

    model_xml = os.path.join(current_directory, '..', 'trained_models', 'medicine', 'optimized_medicine.xml')
    # model_xml = os.path.join(os.path.dirname(__file__), 'DoctorDetailContract.json')
    quantized_model = core.read_model(model_xml)

    quantized_compiled_model = core.compile_model(model=quantized_model, device_name="CPU")
    def pre_process_image(imagePath, img_height=180):
        n, c, h, w = [1, 3, img_height, img_height]
        
        # image = Image.open(BytesIO(imagePath))
        image = imagePath.resize((h, w), resample=Image.BILINEAR)

        image = np.array(image)

        input_image = image.reshape((n, h, w, c))

        return input_image

    input_layer = quantized_compiled_model.input(0)
    output_layer = quantized_compiled_model.output(0)

    class_file = os.path.join(current_directory, '..', 'trained_models', 'medicine', 'medicine_class_list.pkl')

    with open(class_file, 'rb') as file:
        loaded_list = pickle.load(file)
    class_names = loaded_list
   
    input_image = pre_process_image(imagePath=image_file)

    res = quantized_compiled_model([input_image])[output_layer]

    score = tf.nn.softmax(res[0])

    # prediction = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
    #         class_names[np.argmax(score)], 100 * np.max(score)
    #     )
    prediction = class_names[np.argmax(score)]
    return prediction
    # return Success("Success", prediction, 200)


def predictPrescription():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['image']

    core = Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    detection_xml = os.path.join(current_directory, '..', 'trained_models', 'charater_recognition', 'intel', 'horizontal-text-detection-0001', 'FP16',
     'horizontal-text-detection-0001.xml')
    detection_bin = os.path.join(current_directory, '..', 'trained_models', 'charater_recognition', 'intel', 'horizontal-text-detection-0001', 'FP16','horizontal-text-detection-0001.bin')

    detection_model = core.read_model(
            model=detection_xml, weights=detection_bin
        )
    
    detection_compiled_model = core.compile_model(model=detection_model, device_name="CPU")
    detection_input_layer = detection_compiled_model.input(0)

    image = load_image(image_file)

    N, C, H, W = detection_input_layer.shape

    resized_image = cv2.resize(image, (W, H))

    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

    output_key = detection_compiled_model.output("boxes")
    boxes = detection_compiled_model([input_image])[output_key]

    boxes = boxes[~np.all(boxes == 0, axis=1)]
    # /mnt/hdd/projects/codeshift/python_backend/trained_models/charater_recognition/public/text-recognition-resnet-fc/FP16/text-recognition-resnet-fc.xml
    recognition_xml = os.path.join(current_directory, '..', 'trained_models', 'charater_recognition', 'public', 'text-recognition-resnet-fc', 'FP16',
     'text-recognition-resnet-fc.xml')
    recognition_bin = os.path.join(current_directory, '..', 'trained_models', 'charater_recognition', 'public', 'text-recognition-resnet-fc', 'FP16','text-recognition-resnet-fc.bin')

    recognition_model = core.read_model(
        model=recognition_xml, weights=recognition_bin
    )

    recognition_compiled_model = core.compile_model(model=recognition_model, device_name="CPU")

    recognition_output_layer = recognition_compiled_model.output(0)
    recognition_input_layer = recognition_compiled_model.input(0)

    # Get the height and width of the input layer.
    _, _, H, W = recognition_input_layer.shape

    # Calculate scale for image resizing.
    (real_y, real_x), (resized_y, resized_x) = image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Convert the image to grayscale for the text recognition model.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get a dictionary to encode output, based on the model documentation.
    letters = "~0123456789abcdefghijklmnopqrstuvwxyz"

    # Prepare an empty list for annotations.
    annotations = list()
    cropped_images = list()
    # fig, ax = plt.subplots(len(boxes), 1, figsize=(5,15), sharex=True, sharey=True)
    # Get annotations for each crop, based on boxes given by the detection model.
    for i, crop in enumerate(boxes):
        # Get coordinates on corners of a crop.
        (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, crop))
        image_crop = run_preprocesing_on_crop(grayscale_image[y_min:y_max, x_min:x_max], (W, H))

        # Run inference with the recognition model.
        result = recognition_compiled_model([image_crop])[recognition_output_layer]

        # Squeeze the output to remove unnecessary dimension.
        recognition_results_test = np.squeeze(result)

        # Read an annotation based on probabilities from the output layer.
        annotation = list()
        for letter in recognition_results_test:
            parsed_letter = letters[letter.argmax()]

            # Returning 0 index from `argmax` signalizes an end of a string.
            if parsed_letter == letters[0]:
                break
            annotation.append(parsed_letter)
        annotations.append("".join(annotation))
        cropped_image = Image.fromarray(image[y_min:y_max, x_min:x_max])
        cropped_images.append(cropped_image)

    boxes_with_annotations = list(zip(boxes, annotations))
    allPrescription = []
    for cropped_image, annotation in zip(cropped_images, annotations):
        allPrescription.append(predictMedicine(cropped_image))
        print(annotation)
    return Success("Success", allPrescription, 200)



def load_image(image_file) -> np.ndarray:
    
    
    # if path.startswith("http"):
    #     # Set User-Agent to Mozilla because some websites block
    #     # requests with User-Agent Python
    #     response = requests.get(path, headers={"User-Agent": "Mozilla/5.0"})
    #     array = np.asarray(bytearray(response.content), dtype="uint8")
    #     image = cv2.imdecode(array, -1)  # Loads the image as BGR
    # else:
        # image = cv2.imread(image)
    image_data = image_file.read()
        
    # Process the image (e.g., convert to OpenCV format)
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image


def multiply_by_ratio(ratio_x, ratio_y, box):
    return [
        max(shape * ratio_y, 10) if idx % 2 else shape * ratio_x
        for idx, shape in enumerate(box[:-1])
    ]


def run_preprocesing_on_crop(crop, net_shape):
    temp_img = cv2.resize(crop, net_shape)
    temp_img = temp_img.reshape((1,) * 2 + temp_img.shape)
    return temp_img


def questionAnswering():
    question = list(request.json.values())
    

    answer = python_backend.contract.deploy.qa_pipeline(table=python_backend.contract.deploy.medicine_table, query=question)
    return jsonify({
        "status" : "Success",
        "Medicine" : answer["cells"][0]
    })


def predictIndividualMedicine():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['image']

    core = Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))

    model_xml = os.path.join(current_directory, '..', 'trained_models', 'medicine', 'optimized_medicine.xml')
    # model_xml = os.path.join(os.path.dirname(__file__), 'DoctorDetailContract.json')
    quantized_model = core.read_model(model_xml)

    quantized_compiled_model = core.compile_model(model=quantized_model, device_name="CPU")
    def pre_process_image(imagePath, img_height=180):
        n, c, h, w = [1, 3, img_height, img_height]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)

        image = np.array(image)

        input_image = image.reshape((n, h, w, c))

        return input_image

    input_layer = quantized_compiled_model.input(0)
    output_layer = quantized_compiled_model.output(0)

    class_file = os.path.join(current_directory, '..', 'trained_models', 'medicine', 'medicine_class_list.pkl')

    with open(class_file, 'rb') as file:
        loaded_list = pickle.load(file)
    class_names = loaded_list
   
    input_image = pre_process_image(imagePath=image_file)

    res = quantized_compiled_model([input_image])[output_layer]

    score = tf.nn.softmax(res[0])

    prediction = "The prescription submitted  belongs to {} medicine".format(
            class_names[np.argmax(score)]
        )
    return Success("Success", prediction, 200)


def speechToText():
    # print(request.files)
    # if 'speech' not in request.files:
    #     return jsonify({'error': 'No file part'})
    alphabet = " abcdefghijklmnopqrstuvwxyz'~"
    # file_name = request.files['speech']
    file_name = "/home/mktetts/Documents/codeshift/models/speech_to_text/data/recorded-audio.wav"
    audio, sampling_rate = librosa.load(path=str(file_name), sr=16000)
    print("1")
    if max(np.abs(audio)) <= 1:
        audio = (audio * (2**15 - 1))
    audio = audio.astype(np.int16)
    def audio_to_mel(audio, sampling_rate):
        assert sampling_rate == 16000, "Only 16 KHz audio supported"
        preemph = 0.97
        preemphased = np.concatenate([audio[:1], audio[1:] - preemph * audio[:-1].astype(np.float32)])

        # Calculate the window length.
        win_length = round(sampling_rate * 0.02)

        # Based on the previously calculated window length, run short-time Fourier transform.
        spec = np.abs(librosa.core.spectrum.stft(preemphased, n_fft=512, hop_length=round(sampling_rate * 0.01),
                    win_length=win_length, center=True, window=scipy.signal.windows.hann(win_length), pad_mode='reflect'))

        # Create mel filter-bank, produce transformation matrix to project current values onto Mel-frequency bins.
        mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=512, n_mels=64, fmin=0.0, fmax=8000.0, htk=False)
        return mel_basis, spec
    print("1")


    def mel_to_input(mel_basis, spec, padding=16):
        # Convert to a logarithmic scale.
        log_melspectrum = np.log(np.dot(mel_basis, np.power(spec, 2)) + 2 ** -24)

        # Normalize the output.
        normalized = (log_melspectrum - log_melspectrum.mean(1)[:, None]) / (log_melspectrum.std(1)[:, None] + 1e-5)

        # Calculate padding.
        remainder = normalized.shape[1] % padding
        if remainder != 0:
            return np.pad(normalized, ((0, 0), (0, padding - remainder)))[None]
        return normalized[None]
    
    mel_basis, spec = audio_to_mel(audio=audio.flatten(), sampling_rate=sampling_rate)
    print("1")

    audio = mel_to_input(mel_basis=mel_basis, spec=spec)
    core = ov.Core()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    model_xml = os.path.join(current_directory, '..', 'trained_models', 'speech_to_text', 'speech_to_text.xml')
    model = core.read_model(
        model=model_xml
        )
    model_input_layer = model.input(0)
    shape = model_input_layer.partial_shape
    shape[2] = -1
    model.reshape({model_input_layer: shape})
    compiled_model = core.compile_model(model=model, device_name="CPU")
    character_probabilities = compiled_model([ov.Tensor(audio)])[0]
    # Remove unnececery dimension
    character_probabilities = np.squeeze(character_probabilities)
    print("1")

    # Run argmax to pick most possible symbols
    character_probabilities = np.argmax(character_probabilities, axis=1)
    def ctc_greedy_decode(predictions):
        previous_letter_id = blank_id = len(alphabet) - 1
        transcription = list()
        for letter_index in predictions:
            if previous_letter_id != letter_index != blank_id:
                transcription.append(alphabet[letter_index])
            previous_letter_id = letter_index
        return ''.join(transcription)
    transcription = ctc_greedy_decode(character_probabilities)
    print("trasdcc", transcription)
    return Success("Success", transcription, 200)