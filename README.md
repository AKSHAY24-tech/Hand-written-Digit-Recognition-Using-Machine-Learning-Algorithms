# Hand-written-Digit-recognition-using-KNN,RF,CNN.NN,SVM


Digit recognition can be used in identifying digits in forms or cheques.A program for getting the images of digits from cheques was coded in python.
This program first identifies the biggest contour ( the image of the cheque ) in the image of a cheque ( with background ), after identifying the biggest contour the image is cropped according to the biggest contour and then is resized to the necessary dimension.Once the scaling and resizing have been done the position of dates and amount of money is the same for all cheque images ( since the position of these sections within the cheque is the same ). The position was identified and was divided into eight equal parts ( each for date and amount of money ) since all the digits are given equal space.
![image](https://user-images.githubusercontent.com/77917201/218696973-7fd065bd-91f5-42ab-b26c-959591304463.png)
