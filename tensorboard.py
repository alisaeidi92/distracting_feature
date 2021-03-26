##Python Imaging Library is a free and open-source additional library for the Python programming language that adds support
## for opening, manipulating, and saving many different image file formats.

import PIL
import scipy.misc
from io import BytesIO   ##A BytesIO object isn't associated with any real file on the disk. It's just a chunk of memory that behaves like a file does. 
import tensorboardX as tb  ##It has the same API as a file object returned from open (with mode r+b , allowing reading and writing of binary data)
from tensorboardX.summary import Summary
##Google's tensorflow's tensorboard is a web server to serve visualizations of the training progress
##of a neural network, it visualizes scalar values, images, text, etc.; these information are saved as events in tensorflow.
##ensorBoard allows tracking and visualizing metrics such as loss and accuracy, visualizing the model graph, 
##viewing histograms, displaying images and much more.


class TensorBoard(object):
    def __init__(self, model_dir):
        self.summary_writer = tb.FileWriter(model_dir)

    def image_summary(self, tag, value, step):
        for idx, img in enumerate(value):
            summary = Summary()                         ##for creating  image summary for model
            bio = BytesIO()                             ##for converting image to encoded form (binary)

            if type(img) == str:                       ##url to image
                img = PIL.Image.open(img)              ##open the image
            elif type(img) == PIL.Image.Image:           ##Image class of PIL(directly an image)
                pass
            else:
                img = scipy.misc.toimage(img)           ##Takes a numpy array and returns a PIL image.

            img.save(bio, format="png")
            image_summary = Summary.Image(encoded_image_string=bio.getvalue())    ##binary encoded string for images
            summary.value.add(tag=f"{tag}/{idx}", image=image_summary)
            self.summary_writer.add_summary(summary, global_step=step)              ##image summary with tag for model

    def scalar_summary(self, tag, value, step):                 ##to save various summary(tag: action, train-loss,valreward,actor,critic obtained in trainer_rl_typeloss file
        summary= Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)
