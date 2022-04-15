# Tarea Regresión Logística
# Sebastián Galindo
# Carné 15452
# Data Mining y Machine Learning

# Managing imports
import pandas as pd
import numpy


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

    calc_accuracy = (tp + tn) / (tp + tn + fp + fn)
    calc_precision = tp / (tp + fp)
    calc_recall = tp / (tp + fn)

    return tp, tn, fp, fn, calc_accuracy, calc_precision, calc_recall


def calculate_net(data, weights):
    net = weights[0] + weights[1] * data["fixed acidity"] + weights[2] * data["volatile acidity"] \
          + weights[3] * data["citric acid"] + weights[4] * data["residual sugar"] \
          + weights[5] * data["chlorides"] + weights[6] * data["free sulfur dioxide"] \
          + weights[7] * data["total sulfur dioxide"] + weights[8] * data["density"] \
          + weights[9] * data["pH"] + weights[10] * data["sulphates"] \
          + weights[11] * data["alcohol"]
    return net


def gradient_descent(iterations, data, weights, alpha, threshold, fold):
    y = data["quality"]
    iterations_loss = []
    decision = []
    for i in range(iterations):
        decision = []
        print("Fold # ", fold + 1, "Iteration #", i + 1)
        net = calculate_net(data, weights)
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

    return decision, y, iterations_loss, weights


def evaluate(data, weights, threshold):
    y = data["quality"]
    decision = []
    net = calculate_net(data, weights)
    y_hat = (1 / (1 + numpy.exp(-net.array)))

    for j in range(len(y_hat)):
        if y_hat[j] >= threshold:
            decision.append(1)
        else:
            decision.append(0)

    return decision, y


def kfold(data, folds, iterations, weights, alpha, threshold):
    splits = numpy.array_split(data, folds)
    kfold_weights = []
    kfold_a = []  # Kfold accuracy
    kfold_p = []  # Kfold precision
    kfold_r = []  # Kfold recall
    for i in range(len(splits)):
        split_training = pd.DataFrame()
        for j in range(len(splits)):
            iterated_split = splits[j]
            if j != i:
                split_training = split_training.append(iterated_split)
        split_test = splits[i]
        split_decision, split_y, split_iterations_loss, split_weights = gradient_descent(iterations, split_training,
                                                                                         weights, alpha,
                                                                                         threshold, i)
        kfold_weights.append(split_weights)
        split_d, split_y = evaluate(split_test, split_weights, threshold)
        split_tp, split_tn, split_fp, split_fn, split_a, split_p, split_r = calculate_confusion_matrix(
            split_d, split_y)
        kfold_a.append(split_a)
        kfold_p.append(split_p)
        kfold_r.append(split_r)

    kfold_avg_a = sum(kfold_a) / folds
    kfold_avg_p = sum(kfold_p) / folds
    kfold_avg_r = sum(kfold_r) / folds
    kfold_weights = sum(kfold_weights) / folds
    return kfold_avg_a, kfold_avg_p, kfold_avg_r, kfold_weights


# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#
# -----------------------------PROGRAM STARTS HERE-------------------------------------#
# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#

# Inputs
print("Reading inputs...")
input_training_file = pd.read_csv("winequality-training.csv")  # CHANGE PATH HERE
input_test_file = pd.read_csv("winequality-test.csv")  # CHANGE PATH HERE
input_iterations = 10000
input_alpha = 0.1
input_threshold = 0.5
input_weights = numpy.zeros(input_training_file.shape[1])
input_folds = 10

# Shuffle training data
input_data_training = input_training_file.sample(frac=1)

# CODE USED WHEN DATA FILE WAS ONLY ONE FILE
# input_data_training = input_data.sample(frac=0.7)
# input_data_test = input_data.drop(input_data_training.index)

# CODE USED TO GENERATE TRAINING AND TEST FILES
# input_data_training.to_csv('winequality-training.csv', index=False)
# input_data_test.to_csv('winequality-test.csv', index=False)

# Normalize data
print("Normalizing data...")
data_training = normalize_data(input_data_training)
data_test = normalize_data(input_test_file)

# Executing k-fold
print("Calculating kfolds...")
kfold_res_a, kfold_res_p, kfold_res_r, kfold_res_weights = kfold(data_training, input_folds, input_iterations,
                                                                 input_weights, input_alpha, input_threshold)
# Execute gradient descent
# print("Executing gradient decent...")
# training_decision, training_y, iterations_loss, weights = gradient_descent(input_iterations, input_data_70, kfold_weights, input_alpha, input_threshold)

# Evaluate model
print("Evaluating test set...")
evaluate_decision, evaluate_y = evaluate(data_test, kfold_res_weights, input_threshold)

# Calculating confusion matrix
print("Calculating confusion matrix...")
true_positives, true_negatives, false_positives, false_negatives, accuracy, precision, recall = calculate_confusion_matrix(
    evaluate_decision, evaluate_y)

# average_loss = sum(iterations_loss) / input_iterations

print("\n----------------------------------INPUTS-----------------------------------")
print("Iteraciones: ", input_iterations)
print("Alpha: ", input_alpha)
print("Threshold: ", input_threshold)
print("Folds: ", input_folds)
print("\n----------------------MATRIZ DE CONFUSION DEL MODELO-----------------------")
print("Accuracy: ", kfold_res_a)
print("Precision: ", kfold_res_p)
print("Recall: ", kfold_res_r)
print("\n----------------------MATRIZ DE CONFUSION EVALUACION-----------------------")
print("Pesos: ", kfold_res_weights)
print("Verdaderos Positivos: ", true_positives)
print("Verdaderos Negativos: ", true_negatives)
print("Falsos Positivos: ", false_positives)
print("Falsos Negativos: ", false_negatives)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
# print("Average loss: ", average_loss)

# Plotting loss
# plt.plot(range(1, input_iterations + 1), iterations_loss)
# plt.xlabel('Number of Iterations')
# plt.ylabel('Loss')
# plt.title('Loss per iterations')
# plt.show()
