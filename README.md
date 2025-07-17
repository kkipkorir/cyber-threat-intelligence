
# 🛡️ Text-Based Cyber Threat Detection (Simplified Overview)

![Cybersecurity](https://upload.wikimedia.org/wikipedia/commons/7/70/Cybersecurity_image.png)

**Team:** Kelvin Kipkorir, Lucy Mutua, Charles Mutembei, Sharon Aoko, Victor Musyoki

---

## 🔍 What’s This About?

In today's digital world, bad actors use text (emails, messages, reports) to spread malware, scams, or launch attacks. Reading all of it manually is too much work, so we built an intelligent system that can automatically read and flag dangerous messages.

This system uses **machine learning** (a form of artificial intelligence) to tell whether a piece of text is **malicious** (dangerous) or **benign** (harmless).

---

## 🎯 Project Goals

- Detect text-based cyber threats automatically.
- Help security analysts sort through large amounts of data faster.
- Build and train a computer model that learns from real examples of threats.

---

## 📊 The Data

We used a dataset with **19,940 entries** from Kaggle. Each entry includes:

- Raw text from threat intelligence reports.
- Labels showing if the text is about a malware, phishing, location, time, etc.
- Many entries were not labeled — we treated those as *safe* examples.

---

## 🧹 How We Prepared the Data

We cleaned and organized the text so a computer could understand it. This included:

- Making all text lowercase.
- Removing symbols and common “filler” words.
- Breaking sentences into individual words (called tokenization).
- Turning words into numbers (vectors) that computers can learn from.

---

## 🧠 Models We Built

We built several versions of our model to see which worked best:

1. **Baseline Model:** A simple neural network. It performed surprisingly well (95% accurate).
2. **Tuned Model:** Same as above but smarter and better trained. Also about 95% accurate, but with better balance.
3. **LSTM Model:** A more advanced memory-based model — but it didn’t work well in our case.
4. **BiLSTM Model:** A powerful model that reads text forward and backward. It performed great — almost as good as the tuned model.

---

## ✅ Best Results

- Our best models were about **95% accurate**.
- They helped identify cyber threats with high confidence.
- The tuned model had fewer false negatives — a very good thing in cybersecurity.

---

## 📦 Conclusion

This project shows that smart machines can read and classify cyber threat text just like a human expert — only faster. It’s a promising step toward automatic cyber defense systems.

---

## 🛠️ Tools Used

- Python, Pandas, NumPy
- Machine Learning (Keras, TensorFlow)
- Natural Language Processing (NLTK)
- Visualization (Matplotlib, Seaborn)

---

## 📁 File Structure

- **`data/`**: Raw dataset (from Kaggle)
- **`notebook/`**: Code and model experiments (Jupyter)
- **`README.md`**: This summary

---