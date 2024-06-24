from collections import defaultdict
import cv2
import numpy as np

from PIL import Image

from fs import AbstractFileSystem

# ============== We define our face classifier for extracting faces ========


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_padding: int = 10


def detect_faces(img):

    # First convert image to gray image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5)
    )

    del gray_image

    return detected_faces


def compute_coordinates(x, y, w, h):

    # We also try to pad the spaces on the pixels around the face
    # This helps FaceNet compute better embeddings

    # Padding represents the number of pixels

    y1 = y-face_padding if (y-face_padding) >= 0 else 0
    y2 = y+h+face_padding

    x1 = x-face_padding if (x - face_padding) >= 0 else 0
    x2 = x+w+face_padding

    return x1, y1, x2, y2


def extract_faces(img, faces):

    # 10 pixels around point is an acceptable epsilon

    processed_faces = (compute_coordinates(*face) for face in faces)

    return [
        cv2.resize(
            img[y1:y2, x1:x2],
            (160, 160),
            interpolation=cv2.INTER_AREA
        )

        for x1, y1, x2, y2 in processed_faces
    ]


def draw_rectangle_on_face(img_vector, face):

    x1, y1, x2, y2 = compute_coordinates(*face)

    return cv2.rectangle(
        img_vector,
        (x1, y1), (x2, y2),
        color=(255, 0, 0),
        thickness=2
    )


def load_image_from_disk(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Resize the image to the target size (160, 160)
    # image = image.resize((160, 160))

    # Convert the image to a NumPy array
    return np.array(image)


def preprocess_image(image_path):

    if isinstance(image_path, str):
        img = load_image_from_disk(image_path)
    else:
        img = image_path

    img = np.around(np.array(img) / 255.0, decimals=12)

    if len(img.shape) < 4:

        img = np.expand_dims(img, axis=0)

    return img


def load_image_database(*, model=None, fs_repo: AbstractFileSystem):
    """
        Load images from file system, and return it
    """

    print("Loading image database...")

    image_files = fs_repo.list_all(exclude={'incidents', 'correct'})

    print("Images: ", len(image_files))

    database = defaultdict(lambda: [])

    for image_filename in image_files:

        path = "/".join(image_filename.split("/")[:-1])

        img_vector = load_image_from_disk(image_filename)

        face_points = detect_faces(img_vector)
        faces = extract_faces(img_vector, face_points)

        if not faces:
            continue

        # Path may consist of face group and face ID
        # We load the image data into a list on the dtabase
        database[path] += faces

    return database
