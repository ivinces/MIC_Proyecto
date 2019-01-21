from sys import argv
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT

isSignificant = 0.7 #try different values.

filename="woodsaw.wav"
# P: list of probabilities
Result, P, classNames = aT.fileClassification(filename, "svmModel", "svm")
print(Result)
print(P)
print(classNames)
winner = np.argmax(P) #pick the result with the highest probability value.

# is the highest value found above the isSignificant threshhold?
if P[winner] > isSignificant :
    print("File: " +filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner]))
else:
    print("Can't classify sound: " + str(P))
