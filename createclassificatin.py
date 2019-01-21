from pyAudioAnalysis import audioTrainTest as aT
import numpy as np


subdirectories = ["./amenaza/disparos", "./amenaza/incendio","./amenaza/tala"]

print(subdirectories)
aT.featureAndTrain(subdirectories, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmModel", False)

