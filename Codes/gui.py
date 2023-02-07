from tkinter import *  # importing necessary packages
from tkinter import ttk, filedialog
from PIL import ImageTk, Image, ImageGrab
from sklearn import svm
import pandas as pd
import dig_KNN                                                  # python file of the KNN algorithm downloaded as .py
                                                                # from jupyter notebook
import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

last_x, last_y = None, None  # used to save x and y co-ordinates of mouse in canvas
img_no = 0  # number of image to be saved ( used in name )


def show():  # function to show the image
    global img, panel
    img = Image.open(path)  # open the image using it's path
    # img = Image.open("pc5.png")
    img = img.resize((int(img.width / 6), int(img.height / 6)), Image.ANTIALIAS)  # resize and print it as a label
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.place(x=startx + 465, y=starty + 50)


def choose():  # function to select the image
    global path
    path = filedialog.askopenfilename(
        title="Choose The Image")  # open the directory interface and allow the user to choose
    # the image and get the path


def preprocess(img):  # preprocess the image to match our algorithm
    k = img
    k = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)  # coloured to grayscale image
    k = cv2.bitwise_not(cv2.threshold(k, 100, 255, cv2.THRESH_BINARY)[1])  # modifying the image to match our dataset
    resized = cv2.resize(k, [28, 28], interpolation=cv2.INTER_AREA)  # resize to 28x28 size image
    img = np.asarray(resized).reshape(-1)  # returning the pixel array
    return img


def compute():  # function to compute the number
    img = cv2.imread(path)  # read the image
    global Knn, Rf, NN, svm_c
    chosen = List.get()  # get the selected algorithm
    if Knn != None:  # refresh the screen ( remove old recognitions )
        Knn.place_forget()
        Rf.place_forget()
        NN.place_forget()
        svm_c.place_forget()

    colour = "Orange"
    back_colour = "Green"

    # Label for all the Algorithm's output
    Knn = Label(root, text="KNN says the number is :", bg="black", fg=colour, font='sans 9 bold')
    Rf = Label(root, text="Rf says the number is :", bg="black", fg=colour, font='sans 9 bold')
    NN = Label(root, text="NN says the number is :", bg="black", fg=colour, font='sans 9 bold')
    svm_c = Label(root, text="Svm says the number is :", bg="black", fg=colour, font='sans 9 bold')

    if chosen == "KNN":  # Call the chosen algorithm's function
        identified = dig_KNN.classifier(img)
        Knn.config(text=Knn["text"] + "\n\t\t\t" + str(identified))
        Knn.place(x=startx, y=starty + 120)

    if chosen == "Random Forest":
        img = preprocess(img)
        rf = RandomForestClassifier(n_estimators=125, max_depth=100, min_samples_split=20)
        rf.fit(X, Y)
        identified = rf.predict(img.reshape(1, -1))
        Rf.config(text=Rf["text"] + "\n\t\t\t" + str(identified))
        Rf.place(x=startx, y=starty + 120)

    if chosen == "SVM":
        img = preprocess(img)
        Svm = svm.SVC()
        Svm.fit(X, Y)
        identified = Svm.predict(img.reshape(1, -1))
        svm_c.config(text=svm_c["text"] + "\n\t\t\t" + str(identified))
        svm_c.place(x=startx, y=starty + 120)

    if chosen == "NN":
        img = preprocess(img)
        Nn = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(20,), random_state=2, max_iter=30000)
        Nn.fit(X, Y)
        nn = Nn.predict(img.reshape(1, -1))
        NN.config(text=NN["text"] + "\n\t\t\t" + str(nn))
        NN.place(x=startx, y=starty + 120)


def resize_background(event):  # function to resize the background image when
    # the screen's size is changed
    im = cop_im.resize((event.width, event.height))  # resize the background image
    bg_img = ImageTk.PhotoImage(im)  # read the resized image and make it as the background image
    mL.config(image=bg_img)
    mL.image = bg_img


def clear_all():  # function to clear the canvas
    global can
    can.delete("all")


def activate(event):  # function to activate the canvas for drawing
    global last_x, last_y
    can.bind('<B1-Motion>', draw)
    last_x, last_y = event.x, event.y


def draw(event):  # function to draw the lines when the cursor is used to draw on the canvas
    global last_x, last_y
    x, y = event.x, event.y  # get the current x,y co-ordinates of the cursor

    # draw the line from previous recorded x,y co-ordinates (last_x,last_y) to the current x,y co-ordinates (x,y)
    can.create_line((last_x, last_y, x, y), width=8, fill='red', smooth=TRUE, capstyle=ROUND, splinesteps=12)

    # change the last x,y co-ordinates
    last_x, last_y = x, y


def Recognize():  # function to get the image in the canvas
    global img_no
    predict = []
    file = f'image_{img_no}.png'  # name for the image to be saved
    widget = can  # select the canvas as widget

    # get the x,y co-ordinates of the image to be captured ( position of the canavs in the screen )
    x = new.winfo_x() + widget.winfo_x() + root.winfo_x()
    y = new.winfo_y() + widget.winfo_y() + root.winfo_y() + 20
    x1 = x + widget.winfo_width() + 60
    y1 = y + widget.winfo_height() + 20

    ImageGrab.grab().crop((x, y, x1, y1)).save(file)  # get the screen shot of the screen
    image = cv2.imread(f'image_{img_no}.png')  # save the image ( optional )

    # preprocess the captured image for identification
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, th_im = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find the contours in the image ( corresponding to the numbers )
    contours = cv2.findContours(th_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # for every number in the image
    for cnt in contours:
        # get the image of the number alone
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 0, 0), 1)
        top = int(0.05 * th_im.shape[0])
        bottom = top
        left = int(0.05 * th_im.shape[1])
        right = left

        # crop the image of the number
        th_up = cv2.copyMakeBorder(th_im, top, bottom, left, right, cv2.BORDER_REPLICATE)
        roi = th_up[y - top:y + h + bottom, x - left:x + w + right]
        imga = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        imga = np.asarray(imga).reshape(-1)

        # send the processed image into the classifier
        Svm = svm.SVC()
        Svm.fit(X, Y)
        identified = Svm.predict(imga.reshape(1, -1))
        pr = Label(new, text="The number is : " + str(identified))
        pr.place(x=new.winfo_width() / 2, y=new.winfo_height() / 2)
        # get the number
        pred = identified

        # print the number over the image ( optional )
        text = str(pred)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 0.5
        colour = (20, 25, 40)
        thickness = 4
        cv2.putText(image, text, (x, y - 5), font, fontscale, colour, thickness)


def open_new():  # function to open the canvas
    global new, can
    new = Toplevel(root)  # create a new level of screen for canavs
    new.title("Hello !")
    can = Canvas(new, width=200, height=200, bg='white')  # creating the canvas
    can.pack()

    can.bind('<Button-1>', activate)  # button for capturing the drawing and recognizing
    save = Button(new, text="Save and Identify", command=Recognize)
    save.pack()

    clear = Button(new, text="Clear Screen", command=clear_all)  # button for clearing the screen
    clear.pack()


D = pd.read_csv("train_data.csv")  # read the training data and get the required values
D = D.values[:10000]
X = D[:, 1:]
Y = D[:, 0]

# create the screen for GUI
root = Tk()
root.geometry("800x450")
root.title("Welcome to Handwritten digit classifier")

# set the background image and option for it to resize automatically
im = Image.open("backgrounds/bg8.jfif")
cop_im = im.copy()
bg_img = ImageTk.PhotoImage(im)
mL = ttk.Label(root, image=bg_img)
mL.bind('<Configure>', resize_background)
mL.pack(fill=BOTH, expand=YES)

# algorithm options
options = ["KNN", "Random Forest", "SVM", "NN"]

# variables used to place widgets in the screen
starty = 20
startx = 150

# instruction label to select the algorithm
Instruct = Label(root, text="Select An Algorithm :", bg="black", fg="#F8B88B", font='sans 9 bold')
Instruct.place(x=startx, y=starty + 50)

# used to see if Knn label is declared atleast once
Knn = None

# combobox for selecting the algorithm
List = ttk.Combobox(root, value=options)
List.place(x=100 + startx, y=starty + 80)

# Button to choose the image
Choose = Button(root, text="Choose An Image", command=lambda: choose(), bg="orange", fg="#F70D1A", font='sans 8 bold',
                width=30)
Choose.place(x=150 + startx, y=starty)

# Button to proceed to identification of the number
Proceed = Button(root, text="Proceed", command=lambda: compute(), bg="orange", fg="#F70D1A", font='sans 8 bold',
                 width=10)
Proceed.place(x=250 + startx, y=starty + 78)

# Button to show the image
Show = Button(root, text="Show The Image", command=lambda: show(), bg="orange", fg="#F70D1A", font='sans 8 bold',
              width=20)
Show.place(x=startx + 450, y=starty)

# Button to open the canvas
Canv = Button(root, text="Go to Canvas", command=lambda: open_new(), bg="orange", fg="#F70D1A", font='sans 8 bold',
              width=20)
Canv.place(x=startx - 50, y=starty)

root.mainloop()
