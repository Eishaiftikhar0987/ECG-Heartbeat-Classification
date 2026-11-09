# Analysis: Classifying Heartbeats from ECG Signals

All numbers below come from `results/table2_network_training.md`, `table3_lr_choice.md`, `table4_test_results.md` and `history.json`.

## 1. What would a model that always answers "N" score? What does that say about accuracy?

The test file has 18,118 N beats out of 21,892 rows, so a model that always says "N" gets 18,118 / 21,892 = **82.76 %** test accuracy without looking at the signal at all. Our network reaches **97.77 %**, which is a gain of about 15 percentage points - but its **balanced accuracy is only 85.66 %**. Accuracy is dominated by the huge N class: N is recognised 99.29 % of the time while S (70.86 %) and F (67.90 %) are far lower. A single accuracy number therefore hides how badly the rare, clinically important beats are handled, so per-class accuracy, the confusion matrix and balanced accuracy must be reported as well.

## 2. Which class has the lowest per-class accuracy, and what is it mistaken for?

**F (fusion beat)** has the lowest per-class accuracy: 110 of 162 test beats = **67.90 %**. From the confusion matrix, F is most often mistaken for **N** (35 of 162 beats, 21.6 %), then for V (17 beats). (S is a close second at 70.86 %, with 146 of its 556 beats predicted as N.)

A fusion beat is, by definition, produced when a normal beat and a ventricular beat occur at almost the same moment, so its waveform is a mixture of an N beat and a V beat. Parts of it look like a normal beat (narrow peak, normal baseline) and parts look like a V beat (wider, distorted shape), which puts it right at the border between the two classes. In addition F has only 641 training rows (0.7 % of the training file) versus 72,471 for N, so the network sees very few examples from which to learn the difference.

## 3. Do the loss curves show overfitting, underfitting or neither?

For the chosen alpha = 0.5, at the best epoch (25) the training loss is 0.0523 and the validation loss is 0.0680 (accuracy 98.44 % vs 98.12 %). There is a **small gap** (about 0.016) between the two curves, but both fall together during the whole run and the validation loss never turns clearly upward; it only becomes noisy (small bumps around epochs 14-17 and 23-27), which comes from the large learning rate with momentum. The validation accuracy stays within about 0.6 points of the training accuracy in every epoch (0.33 points at the best epoch). So the network is **neither clearly overfitting nor underfitting**: it generalises well, with at most a very mild generalisation gap (over the last ten epochs both curves flatten, the training loss around 0.05 and the validation loss around 0.07, so more epochs would bring little). The curve for alpha = 0.01 was still falling at epoch 30, which shows that this small learning rate simply learns too slowly (underfitting within 30 epochs).

## 4. Why not initialise all weights to zero? Why sqrt(2 / n[l-1]) for ReLU?

With all weights equal (zero), every unit in a layer computes the same value and receives the same gradient, so all units of a layer stay identical forever (the symmetry is never broken) and the layer behaves like a single unit. With zero weights and ReLU it is even worse: the hidden activations are 0, so the gradient reaching the weight matrices is 0 too and nothing but the output bias can learn at the start. Random initial weights break the symmetry.

The scale matters because of how the variance of a signal changes layer by layer: for z = sum of n[l-1] terms w·a, Var(z) = n[l-1] · Var(w) · E[a²]. ReLU sets about half of its inputs to zero, so E[a²] = Var(z_prev) / 2 for a zero-mean symmetric z_prev. To keep Var(z) the same from layer to layer (so that activations and gradients do not shrink to zero or explode in a deeper network) we need n[l-1] · Var(w) / 2 = 1, i.e. Var(w) = 2 / n[l-1] and standard deviation sqrt(2 / n[l-1]). This is the He rule.

## 5. Momentum and weight decay

**Momentum.** The update uses a velocity v = beta·v + (1 - beta)·g, an exponential average of past gradients. If two consecutive mini-batch gradients point in opposite directions, the new gradient partly cancels the stored velocity (for example with beta = 0.9 and g_new = -g_old the velocity changes from g_old to 0.9·g_old - 0.1·g_old = 0.8·g_old), so the step becomes smaller instead of jumping back and forth: momentum **damps oscillations**, while gradients that keep pointing the same way build up speed.

**Weight decay.** For a weight that receives no gradient (dJ/dP = 0) the effective gradient is g = lambda·P, so every update moves it towards zero: P shrinks by about a factor (1 - alpha·lambda) per step (smoothed by the momentum), i.e. it **decays exponentially to 0**. Weights that are not supported by the data are therefore removed, which keeps the weights small and acts as regularisation.

## 6. Helping the rare F class (idea only, not run)

A simple change to the **loss** is a **class-weighted cross-entropy**: multiply the loss of each example by a weight w_y proportional to 1 / (number of training rows of class y) (for example w_F = 72,471 / 641 ≈ 113 relative to N, normalised so the average weight is about 1). A mistake on a rare F beat then costs as much in total as many mistakes on N beats, so the gradient no longer ignores the rare class. (A data-side alternative is to oversample F beats, or to create extra F beats by adding small noise or time shifts to the existing ones, so that the network sees them as often as the other classes.)
