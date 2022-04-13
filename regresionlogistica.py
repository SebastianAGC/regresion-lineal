# Tarea Regresión Logística
# Sebastián Galindo
# Carné 15452
# Data Mining y Machine Learning

# Managing imports
import pandas as pd
import numpy
import matplotlib.pyplot as plt


# Defining functions
def normalize_quality(quality):
    if quality == 'Bad':
        return 0
    else:
        return 1


def normalize_data(data):
    max_fixed_acidity = max(data["fixed acidity"])
    max_volatile_acidity = max(data["volatile acidity"])
    max_citric_acid = max(data["citric acid"])
    max_residual_sugar = max(data["residual sugar"])
    max_chlorides = max(data["chlorides"])
    max_free_sulfur_dioxide = max(data["free sulfur dioxide"])
    max_total_sulfur_dioxide = max(data["total sulfur dioxide"])
    max_density = max(data["density"])
    max_ph = max(data["pH"])
    max_sulphates = max(data["sulphates"])
    max_alcohol = max(data["alcohol"])

    data["fixed acidity"] = data["fixed acidity"] / max_fixed_acidity
    data["volatile acidity"] = data["volatile acidity"] / max_volatile_acidity
    data["citric acid"] = data["citric acid"] / max_citric_acid
    data["residual sugar"] = data["residual sugar"] / max_residual_sugar
    data["chlorides"] = data["chlorides"] / max_chlorides
    data["free sulfur dioxide"] = data["free sulfur dioxide"] / max_free_sulfur_dioxide
    data["total sulfur dioxide"] = data["total sulfur dioxide"] / max_total_sulfur_dioxide
    data["density"] = data["density"] / max_density
    data["pH"] = data["pH"] / max_ph
    data["sulphates"] = data["sulphates"] / max_sulphates
    data["alcohol"] = data["alcohol"] / max_alcohol
    data["quality"] = data["quality"].apply(normalize_quality)
    return data


def calculate_confusion_matrix(calculated_values, real_values):
    tp, tn, fp, fn = 0, 0, 0, 0
    real_values = real_values.tolist()

    for i in range(len(calculated_values)):
        calc = calculated_values[i]
        real = real_values[i]
        if calc == 1 and real == 1:
            tp = tp + 1
        elif calc == 0 and real == 0:
            tn = tn + 1
        elif calc == 1 and real == 0:
            fp = fp + 1
        elif calc == 0 and real == 1:
            fn = fn + 1

    return tp, tn, fp, fn


def gradient_descent(iterations, data, weights, alpha, threshold):
    y = data["quality"]
    iterations_loss = []
    decision = []
    for i in range(iterations):
        decision = []
        # print("Number of iteration: ", i)
        net = weights[0] + weights[1] * data["fixed acidity"] + weights[2] * data["volatile acidity"] \
              + weights[3] * data["citric acid"] + weights[4] * data["residual sugar"] \
              + weights[5] * data["chlorides"] + weights[6] * data["free sulfur dioxide"] \
              + weights[7] * data["total sulfur dioxide"] + weights[8] * data["density"] \
              + weights[9] * data["pH"] + weights[10] * data["sulphates"] \
              + weights[11] * data["alcohol"]
        y_hat = (1 / (1 + numpy.exp(-net.array)))

        gradient0 = y_hat - y
        gradient1 = gradient0 * data["fixed acidity"]
        gradient2 = gradient0 * data["volatile acidity"]
        gradient3 = gradient0 * data["citric acid"]
        gradient4 = gradient0 * data["residual sugar"]
        gradient5 = gradient0 * data["chlorides"]
        gradient6 = gradient0 * data["free sulfur dioxide"]
        gradient7 = gradient0 * data["total sulfur dioxide"]
        gradient8 = gradient0 * data["density"]
        gradient9 = gradient0 * data["pH"]
        gradient10 = gradient0 * data["sulphates"]
        gradient11 = gradient0 * data["alcohol"]

        weights[0] = weights[0] - alpha * numpy.average(gradient0)
        weights[1] = weights[1] - alpha * numpy.average(gradient1)
        weights[2] = weights[2] - alpha * numpy.average(gradient2)
        weights[3] = weights[3] - alpha * numpy.average(gradient3)
        weights[4] = weights[4] - alpha * numpy.average(gradient4)
        weights[5] = weights[5] - alpha * numpy.average(gradient5)
        weights[6] = weights[6] - alpha * numpy.average(gradient6)
        weights[7] = weights[7] - alpha * numpy.average(gradient7)
        weights[8] = weights[8] - alpha * numpy.average(gradient8)
        weights[9] = weights[9] - alpha * numpy.average(gradient9)
        weights[10] = weights[10] - alpha * numpy.average(gradient10)
        weights[11] = weights[11] - alpha * numpy.average(gradient11)

        for j in range(len(y_hat)):
            if y_hat[j] >= threshold:
                decision.append(1)
            else:
                decision.append(0)

        loss = -y * numpy.log(y_hat) - (1 - y) * numpy.log(1 - y_hat)
        iteration_loss = numpy.average(loss)
        iterations_loss.append(iteration_loss)
        # print("Loss: ", iteration_loss)

    return decision, y, iterations_loss


# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#
# -----------------------------PROGRAM STARTS HERE-------------------------------------#
# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#

# Inputs
print("Reading inputs...")
input_data = pd.read_csv("winequality-red-training.csv")  # CHANGE PATH HERE
input_iterations = 10000
input_alpha = 0.2
input_threshold = 0.5
input_weights = numpy.zeros(input_data.shape[1])

# Normalize data
print("Normalizing data...")
input_data = normalize_data(input_data)

# Execute gradient descent
print("Executing gradient decent...")
decision, y, iterations_loss = gradient_descent(input_iterations, input_data, input_weights, input_alpha,
                                                input_threshold)

# Calculating confusion matrix
true_positives, true_negatives, false_positives, false_negatives = calculate_confusion_matrix(decision, y)
accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
average_loss = sum(iterations_loss) / input_iterations

print("\n----------------------------MATRIZ DE CONFUSION----------------------------")
print("Verdaderos Positivos: ", true_positives)
print("Verdaderos Negativos: ", true_negatives)
print("Falsos Positivos: ", false_positives)
print("Falsos Negativos: ", false_negatives)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Average loss: ", average_loss)

# Plotting loss
plt.plot(range(1, input_iterations + 1), iterations_loss)
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.title('Loss per iterations')
plt.show()
