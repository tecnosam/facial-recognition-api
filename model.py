import io
import tensorflow as tf
import numpy as np

from PIL import Image

from utils import (
    load_image_database,
    detect_faces,
    extract_faces,
    preprocess_image,
)

from fs import LocalFileSystem


from settings import MODEL_PATH, RECORDS_DIRECTORY


# ================ We configure the processor to be optimized =============

# Set up tensorflow to use GPU
# devices = tf.config.list_physical_devices('GPU')

# if devices:s
#    tf.config.experimental.set_visible_devices(devices[0], 'GPU')
#    tf.config.experimental.set_memory_growth(devices[0], True)



class ModelAPI:

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    database = {}

    MAXIMUM_EUCLIDEAN_DISTANCE = 0.7

    @staticmethod
    def predict(img):

        embedding = ModelAPI.model(img)
        euclidean_norm = np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)

        # Normalize encodings
        encoding = embedding / euclidean_norm

        # Delete embedding from memory
        del embedding

        return encoding

    @classmethod
    def search(cls, img, return_max: bool = False):

        face_points = detect_faces(img)

        faces = extract_faces(img, face_points)

        inferences = [ModelAPI._search_database_for_face(face) for face in faces]

        if not return_max:
            return face_points, inferences

        if not inferences:
            return None

        distance, identity = min(inferences, key=lambda inf: inf[0])

        if distance > cls.MAXIMUM_EUCLIDEAN_DISTANCE:
            return None

        return identity

    @staticmethod
    def _compute_face_encoding(face):

        img = preprocess_image(face)

        embedding = ModelAPI.predict(img)

        return embedding

    @staticmethod
    def _search_database_for_face(face):

        encoding = ModelAPI._compute_face_encoding(face)

        min_dist = 100
        identity = None

        for (name, db_enc) in ModelAPI.database.items():

            dist = np.linalg.norm(np.subtract(db_enc, encoding), axis=1)

            if np.any(dist < min_dist):
                min_dist = float(dist.min())
                identity = name

        return min_dist, identity

    @staticmethod
    def _register_face(path, image_encoding):

        ModelAPI.database[path] = image_encoding

    @staticmethod
    def load_image_database():

        fs_repo = LocalFileSystem(RECORDS_DIRECTORY)
        database = load_image_database(fs_repo=fs_repo)

        for path, value in database.items():

            # Compute embeddings of all the faces for that path
            arr = np.array(value)
            encodings = ModelAPI._compute_face_encoding(arr)
            ModelAPI._register_face(path, encodings)

        return

    @staticmethod
    def to_image(binary_content: bytes):
        # Convert binary content to a PIL Image
        image = Image.open(io.BytesIO(binary_content)).convert("RGB")

        # Convert PIL Image to a NumPy array
        np_image = np.array(image)
        
        return np_image
