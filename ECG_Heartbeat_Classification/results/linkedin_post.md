🫀 Can a small neural network read heartbeats — and is accuracy enough?

For my Applied Machine Learning assignment I built an ECG heartbeat classifier completely from scratch in PyTorch: my own weight matrices, He initialisation, softmax cross-entropy, and mini-batch gradient descent with momentum and weight decay (no nn.Linear, no torch.optim).

📊 Results on 21,892 unseen test beats (5 classes: N, S, V, F, Q):
• Test accuracy: 97.77 %
• But a model that always says "normal" already gets 82.76 %!
• Balanced accuracy: only 85.66 % — the rare fusion (F) and supraventricular (S) beats are recognised only ~68–71 % of the time.

💡 Lesson: on imbalanced data, always check per-class accuracy and the confusion matrix, not just overall accuracy.

📝 Blog: [add Medium link]
💻 Code: [add GitHub link]

#MachineLearning #PyTorch #DeepLearning #ECG #HealthcareAI #DataScience
