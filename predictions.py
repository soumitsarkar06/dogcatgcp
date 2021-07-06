from PIL import Image
from keras.models import load_model
import numpy, json

class A():
    def __init__(self, filename):
        self.filename = filename
        self.model = load_model("ImageModel.h5")

    def open_file(self):
        with open("class_index.txt", "r") as f:
            self.classes = json.loads(f.read())
            self.classes = {values:keys for keys, values in self.classes.items()}


    def predic_out(self):
        with open(self.filename, "rb") as f:
            image = Image.open(f).resize([64, 64])
            image = numpy.asarray(image)
            image = numpy.expand_dims(image, axis=0)
            pred = self.model.predict(image)
            self.open_file()
            if pred[0][0] > .5:
                return self.classes[1]
            else:
                return self.classes[0]


