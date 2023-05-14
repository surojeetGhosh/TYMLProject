from tkinter import *   # for application framework
from tkinter import filedialog  # for opening window files
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import decode_predictions
import numpy as np

# load model
model = load_model('./models/denseNet11Classes.h5')
classes = {0: 'Bacterial Spot',
           1: 'Early Blight',
           2: 'Late Blight',
           3: 'Leaf Mold',
           4: 'Septoria Leaf Spot',
           5: 'Spider Mites Two Spotted Spider Mite',
           6: 'Target Spot',
           7: 'Tomato Yellow Leaf Curl Virus',
           8: 'Tomato Mosaic Virus',
           9: 'Healthy',
           10: 'Powdery Mildew'}

# class for gui


class GUI:
    def __init__(self, master, h, w, bgColor=None):
        self.master = master
        self.canvas = Canvas(master, height=h, width=w,
                             bg=bgColor)  # canvas screen size
        self.canvas.pack()


filename = None


def selectfile():
    global filename, labelLogger, labelImage
    filename = filedialog.askopenfilename(
        initialdir='/', title='Select a file')
    with open(filename) as f:
        extension = f.name.split(".")[-1]
        if extension.lower() == "png" or extension.lower() == "jpg":
            labelLogger = Label(frameLogger, bg='#ffffff', text='Image Loaded', fg='#617A55',
                                font=('Comic Sans MS', 15, 'bold'))
            labelLogger.place(relheight=1, relwidth=1)
            imageNeedToBePredicted = ImageTk.PhotoImage(Image.open(filename))
            # container for storing elements in this case it is
            labelImage = Label(canvasImage, image=imageNeedToBePredicted)
            labelImage.place(x=0, y=0, relheight=1, relwidth=1)
            # used for storing images
            # relative position
            self.label.place(x=0, y=0, relwidth=1, relheight=1)
            return
        else:
            labelLogger = Label(frameLogger, bg='#ffffff', text=f'File is not supported', fg='#617A55',
                                font=('Comic Sans MS', 15, 'bold'))
            labelLogger.place(relheight=1, relwidth=1)
            return


def predict():
    global filename, labelLogger
    if filename is None:
        labelLogger = Label(frameLogger, bg='#ffffff', text=f'Select a image', fg='#617A55',
                            font=('Comic Sans MS', 15, 'bold'))
        labelLogger.place(relheight=1, relwidth=1)
    else:
        img = image.load_img(filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predict = model.predict(img_array)
        predictedClassIndex = np.argmax(predict)
        predictedClass = classes[predictedClassIndex]
        probability = np.max(predict)
        print(predictedClass, probability)
        message = "Predicted Class: " + predictedClass + \
            "\nProbability: " + str(probability)
        labelLogger = Label(frameLogger, bg='#ffffff', text=message, fg='#617A55',
                            font=('Comic Sans MS', 15, 'bold'))
        labelLogger.place(relheight=1, relwidth=1)

        filename = None


# tkinter window
root = Tk()   # for initiating tkinter window
root.title("Tomato Leaf Disease Detection")   # title
a = GUI(root, 600, 500, "#617A55")   # for background screen setup
frameHeading = Frame(root, bg='#FFF8D6', bd=5)  # for styling purpose
frameHeading.place(relx=0.1, rely=0.1, relheight=0.1, relwidth=0.8)
labelHeading = Label(frameHeading, bg='#FFF8D6', text='Detect Tomato Leaf Disease', fg='#99A98F',
                     font=('Comic Sans MS', 8, 'bold'))  # for heading display
labelHeading.place(relheight=1, relwidth=1)

canvasImage = Canvas(root, bg='DarkGoldenrod1',
                     bd=3)  # selecting game gestures
canvasImage.place(relx=0.1, rely=0.22, relheight=0.35, relwidth=0.8)
imageNeedToBePredicted = ImageTk.PhotoImage(Image.open("white.jpg"))
# container for storing elements in this case it is
labelImage = Label(canvasImage, image=imageNeedToBePredicted)
labelImage.place(relheight=1, relwidth=1)

frameOption = Frame(root, bg='DarkGoldenrod1', bd=3)  # selecting game gestures
frameOption.place(relx=0.35, rely=0.9, relheight=0.1, relwidth=0.3)
select = Button(frameOption, text='SELECT IMAGE', fg='sienna2', bg='Antiquewhite2',
                font=('Comic Sans MS', 7, 'bold'), command=selectfile)
select.place(relheight=0.5, relwidth=1)

predict = Button(frameOption, text='Predict', fg='sienna2', bg='Antiquewhite2',
                 font=('Comic Sans MS', 7, 'bold'), command=predict)
predict.place(rely=0.5, relheight=0.5, relwidth=1)

frameLogger = Frame(root, bg='#FFF8D6', bd=5)  # for showing error
frameLogger.place(relx=0.1, rely=0.6, relheight=0.3, relwidth=0.8)
labelLogger = Label(frameLogger, bg='#ffffff', text='', fg='#FFF8D6',
                    font=('Comic Sans MS', 15, 'bold'))
labelLogger.place(relheight=1, relwidth=1)


root.mainloop()
