import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import RMSprop
from matplotlib import pyplot as plt


def solution(dim,firstLayerNeurons,neuronNumbers, trainingFile, testingFile, reg ,learning_rate ,epochNumber):


    train_data, train_labels, validation_data, validation_labels = acquisition(trainingFile)

    test_Data, test_labels = acquisition(testingFile)

    # Define Model
    # input_dim 5 since we have 5 attributes
    initializer = tf.keras.initializers.HeNormal(seed=10)

    zeroInitializer = tf.keras.initializers.Zeros()

    model = tf.keras.Sequential(
        [
            layers.Dense(firstLayerNeurons, input_dim = dim, activation='relu', kernel_initializer=initializer,
                         bias_initializer=zeroInitializer, activity_regularizer=tf.keras.regularizers.l2(reg)),
            layers.Dense(neuronNumbers, activation='relu', kernel_initializer=initializer,
                         bias_initializer=zeroInitializer, activity_regularizer=tf.keras.regularizers.l2(reg)),
            layers.Dense(neuronNumbers, activation='relu', kernel_initializer=initializer,
                         bias_initializer=zeroInitializer, activity_regularizer=tf.keras.regularizers.l2(reg)),
            layers.Dense(neuronNumbers, activation='relu', kernel_initializer=initializer,
                         bias_initializer=zeroInitializer, activity_regularizer=tf.keras.regularizers.l2(reg)),


            layers.Dense(400, activation='softmax')
        ]
    )

    # compile Model
    opt = RMSprop(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.save_weights('model.h5')
    model.summary()

    #earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=2, restore_best_weights=True)
    #history = model.fit(X_train, y_train, batch_size=1144, epochs=250, validation_data=(X_test, y_test), callbacks=[earlystopping])

    from operator import add
    finalLoss = 0
    finalAccuracy=0
    tempValLoss = []
    tempLoss = []
    tempAccuracy = []
    tempValAccuracy = []
    iterations = 10
    #validation_data=(X_test, y_test)
    for i in range(iterations):
        model.load_weights('model.h5')
        history = model.fit(train_data,train_labels, batch_size=100, epochs=epochNumber, validation_data=(validation_data, validation_labels))
        if i==0:
            tempLoss = history.history['loss']
            tempValLoss = history.history['val_loss']
            tempAccuracy = history.history['accuracy']
            tempValAccuracy = history.history['val_accuracy']
        else:
            tempLoss = list(map(add, history.history['loss'], tempLoss))
            tempValLoss = list(map(add, history.history['val_loss'], tempValLoss))
            tempAccuracy = list(map(add, history.history['accuracy'], tempAccuracy))
            tempValAccuracy = list(map(add, history.history['val_accuracy'], tempValAccuracy))


        evLoss, evAccuracy = model.evaluate(test_Data, test_labels)
        finalLoss = finalLoss+evLoss
        finalAccuracy = finalAccuracy + evAccuracy

    tempLoss[:] = [x / iterations for x in tempLoss]
    tempValLoss[:] = [x / iterations for x in tempValLoss]
    tempAccuracy[:] = [x / iterations for x in tempAccuracy]
    tempValAccuracy[:] = [x / iterations for x in tempValAccuracy]

    finalAccuracy = finalAccuracy/iterations
    finalLoss = finalLoss/iterations
    print('Accuracy: ' + str(finalAccuracy * 100))
    print('Loss: ' + str(finalLoss))


    plot1 = plt.figure(1)
    plt.plot(tempLoss)
    plt.plot(tempValLoss)
    plt.title('Loss - reg= ' + str(reg) + ' lr= ' + str(learning_rate) + ' - 3 Hidden Layers')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.text(250, tempLoss[-1], str(tempLoss[-1]))
    plt.text(250, tempValLoss[-1], str(tempValLoss[-1]))

    plot2 = plt.figure(2)
    plt.plot(tempAccuracy)
    plt.plot(tempValAccuracy)
    plt.title('Accuracy - reg = '+ str(reg) + ' lr= ' + str(learning_rate) + ' - 3 Hidden Layers')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.text(250, tempAccuracy[-1], str(tempAccuracy[-1]))
    plt.text(250, tempValAccuracy[-1], str(tempValAccuracy[-1]))
    plt.show()

def acquisition(inputFiles):
    features = []
    classes = []
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    loopCounter = 0
    first = []
    keyword = ['@RELATION', '@ATTRIBUTE', '@DATA']
    for file in inputFiles:
        features = []
        classes = []
        validation_data = []
        validation_labels = []
        classChangeIndex = []
        lineCounter = 1
        tempClass = 1
        with open(file) as textFile:
            for line in textFile:
                if not any(i in line for i in keyword):
                    text = [i.strip() for i in line.split(',')]
                    if len(text) > 1:
                        features.append(text[:len(text) - 1])
                        classes.append([text[len(text) - 1]])
                        if (int(text[len(text) - 1]) != tempClass):
                            classChangeIndex.append(lineCounter - 1)
                            tempClass = int(text[len(text) - 1])
                        lineCounter += 1

        if len(first) == 0:
            first = features

    if len(inputFiles) > 1:
        concat = []
        for i in range(len(first)):
            a = first[i]
            b = features[i]
            x = a + b
            concat.append(x)

        features = concat


    if(file == 'IrisGeometicFeatures_TestingSet.txt' or file == 'IrisTextureFeatures_TestingSet.txt'):

        features = np.array(features, dtype=np.float32)
        classes = np.array(classes, dtype=np.uint8)
        # Class labels must start from 0 for to_categorical function
        # one hot encoding keras
        classes = [x - 1 for x in classes]
        classes = tf.keras.utils.to_categorical(classes, num_classes=3, dtype=np.uint8)

        # normalization
        sc = StandardScaler()
        features = sc.fit_transform(features)

        return [features, classes]

    if(file == 'IrisGeometicFeatures_TrainingSet.txt' or file == 'IrisTextureFeatures_TrainingSet.txt'):

        if (int(len(features[:classChangeIndex[0]])) % 5 == 0):
            size1 = int(len(features[:classChangeIndex[0]]) * 20 / 100)
        else:
            size1 = int(len(features[:classChangeIndex[0]]) * 20 / 100) + 1
        if (int(len(features[classChangeIndex[0] + 1:classChangeIndex[1]]) % 5 == 0)):
            size2 = int(len(features[classChangeIndex[0] + 1:classChangeIndex[1]]) * 20 / 100)
        else:
            size2 = int(len(features[classChangeIndex[0] + 1:classChangeIndex[1]]) * 20 / 100) + 1
        if int(len(features[classChangeIndex[1] + 1:len(features)]) % 5 == 0):
            size3 = int(len(features[classChangeIndex[1] + 1:len(features)]) * 20 / 100)
        else:
            size3 = int(len(features[classChangeIndex[1] + 1:len(features)]) * 20 / 100) + 1



        tempFeatures = features[:classChangeIndex[0] - size1]
        tempFeatures.extend(features[(classChangeIndex[0] + 1):(classChangeIndex[1] - size2)])
        tempFeatures.extend(features[classChangeIndex[1] + 1:len(features) - size3])

        tempValidationFeatures = features[classChangeIndex[0] - size1 + 1:classChangeIndex[0]]
        tempValidationFeatures.extend(features[classChangeIndex[1] - size2 + 1:classChangeIndex[1]])
        tempValidationFeatures.extend(features[len(features) - size3 + 1:len(features)])

        tempClasses = classes[:classChangeIndex[0] - size1]
        tempClasses.extend(classes[classChangeIndex[0] + 1:classChangeIndex[1] - size2])
        tempClasses.extend(classes[classChangeIndex[1] + 1:len(classes) - size3])

        tempValidationClasses = classes[classChangeIndex[0] - size1 + 1:classChangeIndex[0]]
        tempValidationClasses.extend(classes[classChangeIndex[1] - size2 + 1:classChangeIndex[1]])
        tempValidationClasses.extend(classes[len(classes) - size3 + 1:len(classes)])
        validation_data = tempValidationFeatures
        validation_labels= (tempValidationClasses)
        train_data = tempFeatures
        train_labels =tempClasses
    if (file == 'IrisGeometicFeatures_TrainingSet.txt' or file == 'IrisTextureFeatures_TrainingSet.txt'):
        #print(train_data[:914])
        train_data = np.array(train_data, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.uint8)

        validation_data = np.array(validation_data, dtype=np.float32)

        validation_labels = np.array(validation_labels, dtype=np.uint8)

        #Class labels must start from 0 for to_categorical function
        train_labels = [x - 1 for x in train_labels]
        validation_labels = [x - 1 for x in validation_labels]
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=3, dtype=np.uint8)
        validation_labels = tf.keras.utils.to_categorical(validation_labels, num_classes=3, dtype=np.uint8)

        sc = StandardScaler()
        validation_data = sc.fit_transform(validation_data)

        sc1 = StandardScaler()
        train_data = sc1.fit_transform(train_data)

        return [train_data, train_labels, validation_data, validation_labels]




def main():

    choose = int(input("Please choose Geometric(1), Texture(2), Both(3):"))
    if(choose == 1):
        solution(5, 5, 4, ['IrisGeometicFeatures_TrainingSet.txt'], ['IrisGeometicFeatures_TestingSet.txt'],0.75 , 0.01 , 250)
    elif (choose == 2):
        solution(9600, 4, 2, ['IrisTextureFeatures_TrainingSet.txt'], ['IrisTextureFeatures_TestingSet.txt'],0.75, 0.01 , 150)
    else:
        solution(9605, 6, 4, ['IrisGeometicFeatures_TrainingSet.txt', 'IrisTextureFeatures_TrainingSet.txt'], ['IrisTextureFeatures_TestingSet.txt', 'IrisGeometicFeatures_TestingSet.txt'], 0.1
    ,0.005 , 150)






if __name__ == '__main__':
    main()
