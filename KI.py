import math
from typing import List
from prettytable import PrettyTable
from sympy import *

roundNu = 4
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def signum(a: int) -> int:
    """
    Signum Function
    :param a: Zahl
    :return: 1 für positve, 0 für negative a Werte
    """
    return 1 if a >= 0 else 0


# For a specific value for an attribute, add all positive outcomes(p) and negative outcomes(n)
def entro(p: int, n: int):
    """
    Entropie für auftrittswahrscheinlichkeit eines attributes berechnen
    :param p: Anzahl der positive Werte
    :param n: Anzahl der negative Werte
    """
    if p == 0 or n == 0:
        return 0
    s = p + n
    out = round(-(p/s) * math.log2(p/s) - (n/s) * math.log2(n/s), roundNu)
    print(f"E({p}, {n}) = -({p}/{s}) * ld({p}/{s}) - ({n}/{s}) * ld({n}/{s}) "
          f"= {out}")
    return out


# gain([(1,3),(1,1)]) for an attribute with two states and two possible outcomes
def gain(s: List[tuple]):
    """
    Beispiel Probelklausur B, Aufgabe 4
    Beispiel Gain von Meer:
    Ja (3,2)
    Nein (1,2)
    :param s: [(3,2),(1,2)]
    :return:
    """
    p = 0
    n = 0
    for i in s:
        p += i[0]
        n += i[1]
    size = p + n
    gain1 = 0
    stringbuilder = ""
    outer_entropy_value = entro(p, n)
    for i in s:
        entropy_value = entro(i[0], i[1])
        gain1 += sum(i) / size * entropy_value
        stringbuilder += f"{sum(i)}/{size} * {entropy_value} + "
    print(f"Gain = {outer_entropy_value} - ({stringbuilder}\b\b\b) = {round(outer_entropy_value - gain1, roundNu)}")


# hmm((0.8,0.2),[(0.6,0.4),(0.5,0.5)],[(0.2,0.6),(0.2,0.6),(0.4,0.2),(0.2,0.6)])
def hmm(initial: tuple, trans: List[tuple], obs: List[tuple]):
    """
    Hidden Markov Model
    :param initial: Initiale Wahrscheinlichkeitswerte von Startpunkt zu erstem Knoten
    :param trans: Transitionswahrscheinlichkeiten (X->X, X->Y, Y->X, Y->Y)
    :param obs: Observations, z.B. Werte für A-A-C

    Beispielwerte Aus Probeklausur C, Aufgabe 3
    hmm(
        (0.8,0.2),                                     - Initialwahrscheinlichkeiten aus Text-Aufgabenstellung
        [(0.6,0.4), (0.5,0.5)],                        - Transitionsmatrix (Tabelle A)
        [(0.2,0.6), (0.2,0.6), (0.4,0.2), (0.2,0.6)]   - beobachtete folge hier für Teilaufgabe b: A-A-C-A
    )
    """
    result = []
    k11 = round(initial[0] * obs[0][0], roundNu)
    k21 = round(initial[1] * obs[0][1], roundNu)
    print(f"a1x = {initial[0]} * {obs[0][0]} = {k11}")
    print(f"a1y = {initial[1]} * {obs[0][1]} = {k21}")
    result.append((k11, k21))

    for i in range(0, len(obs)-1):
        kn1 = round(result[i][0] * obs[i+1][0] * trans[0][0]\
            + result[i][1] * obs[i+1][0] * trans[1][0], roundNu)
        kn2 = round(result[i][1] * obs[i+1][1] * trans[1][1]\
            + result[i][0] * obs[i+1][1] * trans[0][1], roundNu)
        print(f"a{i+2}x = {result[i][0]} * {obs[i+1][0]} * {trans[0][0]} " +
              f"+ {result[i][1]} * {obs[i+1][0]} * {trans[1][0]} = {kn1} ")
        print(f"a{i+2}y = {result[i][0]} * {obs[i + 1][1]} * {trans[1][1]} " +
              f"+ {result[i][1]} * {obs[i + 1][1]} * {trans[0][1]} = {kn2} ")
        result.append((kn1, kn2))

    result.append(round(sum(result[-1:][0]), roundNu))
    print(result)
    return result


def perceptronLearn(inVector: list, weights: list, outExp, threshold: int = 0):
    '''
    perceptronLearn([1,0,1,1],[0.2,0.7,0.9,-0.7],0), Prbeklausur C, Aufgabe 4.a
    :param inVector: X Werte (Eingänge): [1,0,1,1]
    :param weights: weights: [0.2,0.7,0.9,-0.7]
    :param outExp: Erwarteter y Wert
    :param threshold: 0
    :return:
    '''
    outReal = 0
    for i in range(len(weights)):
        outReal += inVector[i] * weights[i]
    outReal -= threshold
    outReal = signum(outReal)
    for i in range(len(weights)):
        new_value = round(weights[i] + inVector[i] * (outExp - outReal), roundNu)
        print(f"w{i}: {weights[i]} + {inVector[i]} * ({outExp} - {outReal}) = {new_value}")
        weights[i] = new_value

    threshold = round(threshold - (outExp - outReal), roundNu)
    if outReal == outExp:
        print("Nothing new to learn")
    print("outReal:", outReal)
    print("Threshold: ", threshold)
    print("Weights: ", weights)


def perceptronOutput(inVector: list, weights: list, threshold: float = 0, jump_fct: callable = signum):
    out = 0
    print("ȳ = T(", end="")
    # iteriere durch alle gewichte/eingabewerte und addiere das produkt der beiden zu out
    for i in range(len(weights)):
        new_value = inVector[i] * weights[i]
        print(f"{inVector[i]} * {weights[i]} + ", end="")
        out += new_value
    # zuletzt wird der Schwellwert abgezogen
    out -= threshold
    print(f"\b\b\b - {threshold}) = T({round(out,roundNu)}) = {jump_fct(out)}")


def hebbLearn(inVector: list, outVector: list, weights: List[List[float]] = None):
    """
    hebbLearn([0,1,0],[0,1],[[-0.4,-0.7,-0.3],[-0.5,0.1,0.3]])
    :param inVector:
    :param outVector:
    :param weights:
    :return:
    """
    if not weights:
        weights = []
        for _ in range(len(outVector)):
            weights.append([0 for _ in range(len(inVector))])

    print(f"x = {inVector}\ny = {outVector}")

    for i in range(len(outVector)):
        for j in range(len(weights[i])):
            new_weight = round(weights[i][j] + outVector[i] * inVector[j], roundNu)
            # transform i and j to indicies
            printable_i = str(i+1).translate(SUB)
            printable_j = str(j+1).translate(SUB)

            print(f"w{printable_j}{printable_i}⁽¹⁾ "
                  f"= w{printable_j}{printable_i}⁽⁰⁾ + y{printable_i} * x{printable_j} "
                  f"= {round(weights[i][j], roundNu)} + {round(outVector[i], roundNu)}"
                  f" * {round(inVector[j], roundNu)} = {round(new_weight, roundNu)}")
            weights[i][j] = new_weight
    print(weights)
    return weights

#Klausur B aufgabe 3a das hier: `hebbOutput((1,-1),[(0.5,0.8),(0.3,0.2),(-0.4,0.5)])`
def hebbOutput(inVector: list, weights: list):
    outVector = []
    for _ in range(len(weights)):
        outVector.append(0)
    for i in range(len(outVector)):
        # transform i and j to indicies
        printable_i = str(i + 1).translate(SUB)
        print(f"y{printable_i} = T(", end="")
        for j in range(len(inVector)):
            new_value = round(outVector[i] + inVector[j] * weights[i][j], roundNu)

            printable_j = str(j + 1).translate(SUB)
            print(f"{weights[i][j]} * {inVector[j]} + ", end="")

            outVector[i] = new_value
        print(f"\b\b\b) = T({outVector[i]}) = {signum(outVector[i])}")

def checkOrtho(data: list):
    print(math.sqrt(sum(map(lambda x: x**2, data))))


def naiveBayes(prob: List[str], laPlace: int = 0):
    """
    Wahrscheinlichkeiten für Naive Bayes bestimmen.
    :param prob: Kommaseperierte Strings für jeden Fall als Liste
    :param laPlace: V-Wert, falls glättung durchgeführt werden soll - standard: 0

    Beispiel für Prob (Probeklausur C, Aufgabe 5):
    ["w,so,sc,n","w,so,st,n","w,r,sc,j","k,r,st,j","w,so,sc,n",
    "k,so,sc,j","w,so,st,j","w,r,st,j","w,r,sc,j","w,so,sc,n"]
    """
    trueLaPlace = bool(laPlace)
    prob: List[List[str]] = [s.lower().split(",") for s in prob]
    measuredResults: List[str] = [i[-1:][0] for i in prob]
    expectedResults: List[str] = []

    # Eingangstabelle ausgeben, um zu validieren das man sich nicht vertippt hat
    in_table = PrettyTable()
    in_table.header = False
    for i, row in enumerate(prob):
        in_table.add_column("irrelevant", [i+1, *row])
    print(in_table)

    table = PrettyTable()
    table.field_names = [" ", "P(Gesamt=Ja)", "P(Gesamt=Nein)", "Naive Bayes"]
    table.align["P(Gesamt=Ja)"] = "r"
    table.align["P(Gesamt=Nein)"] = "r"

    yesProbability = sum(1 for i in prob if i[-1:][0] == "j") / len(prob)
    noProbability = 1 - yesProbability

    print(f"P(Gesamt=j) = {yesProbability}")
    print(f"P(Gesamt=n) = {noProbability}")

    yesList = [i for i in prob if i[-1:][0] == "j"]
    noList = [i for i in prob if i[-1:][0] == "n"]

    for ctr, i in enumerate(prob):
        multiListYes = []
        multiListNo = []
        for attribute in i[:-1]:
            yesCount = 0
            noCount = 0
            for case in yesList:
                if case.count(attribute):
                    yesCount += 1
            for case in noList:
                if case.count(attribute):
                    noCount += 1
            multiListYes.append(yesCount + trueLaPlace)
            multiListNo.append(noCount + trueLaPlace)
        yes = yesProbability
        no = noProbability
        for i in multiListYes:
            yes *= (i / (len(yesList) + laPlace))
        for i in multiListNo:
            no *= (i / (len(noList) + laPlace))
        expectedResults.append("j" if yes > no else "n")
        table.add_row([
            ctr + 1,
            f"{yesProbability} * {' * '.join([str(round(s, roundNu)) + f'/{(len(yesList) + laPlace)}' for s in multiListYes])} = {round(yes, roundNu)}",
            f"{noProbability} * {' * '.join([str(round(s, roundNu)) + f'/{(len(noList) + laPlace)}' for s in multiListNo])} = {round(no, roundNu)}",
            "Ja" if yes > no else "Nein"
        ])
    print(table)
    print(measuredResults)
    print(expectedResults)
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    for i in range(len(measuredResults)):
        if measuredResults[i] == "j" and expectedResults[i] == "j":
            truePositive += 1
        if measuredResults[i] == "n" and expectedResults[i] == "j":
            falsePositive += 1
        if measuredResults[i] == "n" and expectedResults[i] == "n":
            trueNegative += 1
        if measuredResults[i] == "j" and expectedResults[i] == "n":
            falseNegative += 1

    precision = round(truePositive / (truePositive + falsePositive), roundNu)
    recall = round(truePositive / (truePositive + falseNegative), roundNu)
    accuracy = round((truePositive + trueNegative) / (truePositive + falsePositive + trueNegative + falseNegative), roundNu)
    print(f"Precision {precision}")
    print(f"Recall {recall}")
    print(f"Accuracy {accuracy}")
    print(f"F-Measure {round((2 * precision * recall)/(precision + recall), roundNu)}")

"""
Functions are currently deprecated, please consider using gradient_whole methode

def gradient_descent_forward(x1, x2, w1, w2,
                             y: int = 0, threshold: int = 0, fct: callable = lambda x: 1/(1 + exp(-x))):
    '''
    gradient_descent_forward(
        gradient_descent_forward(1, 1, 0.5, 0.4),
        gradient_descent_forward(1, 1, 0.9, 1), 
        -1.2, 1.1
    )

    :param x1:
    :param x2:
    :param w1:
    :param w2:
    :param y:
    :param threshold:
    :param fct:
    :return:
    '''
    # TODO threshold
    value = N(fct(x1 * w1 + x2 * w2))
    print(f"fct({x1} * {w1} + {x2} * {w2}) = fct({x1 * w1} + {x2 * w2}) = fct({x1 * w1 + x2 * w2}) = {value}")
    mse = round(-(y - value), roundNu)
    print(f"e(Mean Square Error) = -({y} - N(fct({x1} * {w1} + {x2} * {w2})) = -({y} - {value}) = {mse}")
    mbe = 0
    print(f"e(Mean Bias Error) = ??? {mbe} ???")
    mte = 0
    print(f"e(Mean Total Error) = ??? {mte} ???")
    return mse

# lambda x : diff(1/(1+exp))


def gradient_descent_backward(x1, x2, w1, w2, y,
                              fct: callable = lambda x: 1/(1 + exp(-x)), diff_fct: callable = lambda x: x * (1 - x)):

    if not diff_fct:
        # BEGIN DEAD CODE, WHEN diff_fct not overwritten
        x = symbols("x", real=True)
        base = diff(fct(x))
        print(base)
        x = y
        base = eval(str(N(base)))*y
        # END DEAD CODE, WHEN diff_fct not overwritten
    else:
        base = N(diff_fct(y)) * y
    print(base)
    ret = base * w1, base * x1, base * w2, base * x2
    print(f"base * w1, base * x1, base * w2, base * x2"
          f"{base} * {w1}, {base} * {x1}, {base} * {w2}, {base} * {x2} "
          f"= {base * w1}, {base * x1}, {base * w2}, {base * x2}")
    return ret

"""

def gradient_whole(x1, x2, w13, w14, w23, w24, w35, w45, yExp, learnrate, errorMethode = "MSE",
                   fct :callable = lambda x : 1/(1 + exp(-x)), diff_fct: callable = lambda x : 1/(1 + exp(-x)) * (1 - 1/(1 + exp(-x)))):
    """
        Übungsblatt 6, Aufgabe 4
        gradient_whole(1,1,0.5,0.9,0.4,1,-1.2,1.1,0,0.1)
    :param x1: 1
    :param x2: 1
    :param w13: 0.5
    :param w14: 0.4
    :param w23: 0.9
    :param w24: 1
    :param w35: -1.2
    :param w45: 1.1
    :param yExp: 0
    :param learnrate: 0.1
    :param errorMethode: "MSE"
    :param fct:
    :param diff_fct:
    :return:
    """
    y1 = x1 * w13 + x2 * w23
    y2 = x1 * w14 + x2 * w24
    y1Sig = fct(y1)
    y2Sig = fct(y2)
    yReal = y1Sig * w35 + y2Sig * w45
    yRealSig = fct(yReal)
    if errorMethode == "MSE":
        error = -1 * (yExp - yRealSig)
    else:
        pass
    errorBack = error * (1-error) * yRealSig
    y1Back = errorBack * w35
    y2Back = errorBack * w45
    w35new = w35 + -1 * learnrate * errorBack * y1Sig
    w45new = w45 + -1 * learnrate * errorBack * y2Sig
    y1BackSig = y1Sig * (1 - y1Sig) * y1Back
    y2BackSig = y2Sig * (1 - y2Sig) * y2Back
    w13new = w13 + -1 * learnrate * y1BackSig
    w23new = w23 + -1 * learnrate * y1BackSig
    w14new = w14 + -1 * learnrate * y2BackSig
    w24new = w24 + -1 * learnrate * y2BackSig
    print(f"y1: {x1} * {w13} + {x2} * {w23} = {y1}")
    print(f"y1: {x1} * {w14} + {x2} * {w24} = {y2}")
    print(f"y1 nach Signum: {round(y1Sig, roundNu)}")
    print(f"y2 nach Signum: {round(y2Sig, roundNu)}")
    print(f"Result vor Signum: {round(y1Sig, roundNu)} * {w35} + {round(y2Sig, roundNu)} * {w45} = {round(yReal, roundNu)}")
    print(f"Result nach Signum: {round(yRealSig, roundNu)}")
    if errorMethode == "MSE":
        print(f"Fehler: -({yExp} -{round(yRealSig, roundNu)}) = {round(error, roundNu)}")
    print(f"Backwert: {round(error, roundNu)} *  (1 - {round(error, roundNu)}) * {round(yRealSig, roundNu)} = {round(errorBack, roundNu)}")
    print(f"y1Back: {round(errorBack, roundNu)} *  {w35} = {round(y1Back, roundNu)}")
    print(f"y1Back: {round(errorBack, roundNu)} *  {w45} = {round(y2Back, roundNu)}")
    print(f"w35 Neu: {w35} + -{learnrate} * {round(errorBack, roundNu)} * {round(y1Sig, roundNu)} = {round(w35new, roundNu)}")
    print(f"w45 Neu: {w45} + -{learnrate} * {round(errorBack, roundNu)} * {round(y2Sig, roundNu)} = {round(w45new, roundNu)}")
    print(f"w13 Neu: {w13} + -{learnrate} * {round(y1BackSig, roundNu)} = {round(w13new, roundNu)}")
    print(f"w23 Neu: {w23} + -{learnrate} * {round(y1BackSig, roundNu)} = {round(w23new, roundNu)}")
    print(f"w14 Neu: {w14} + -{learnrate} * {round(y2BackSig, roundNu)} = {round(w14new, roundNu)}")
    print(f"w24 Neu: {w24} + -{learnrate} * {round(y2BackSig, roundNu)} = {round(w24new, roundNu)}")