DAT158 – SMS Spam Detection

Project Description:
This project is a machine learning-based web application that classifies SMS messages as either spam or ham (legitimate messages).
The application was developed using Python, Flask and scikit-learn. Users can enter an SMS message and receive a prediction through a simple website.

Dataset:
The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.
The dataset contains 5,572 labeled SMS messages:
- 4,825 ham messages
- 747 spam messages

Machine Learning Models:
Two classification models were trained and evaluated:
- Multinomial Naive Bayes
- Logistic Regression
Both models use TF-IDF for text vectorization.
Logistic Regression was selected for the final application because it achieved better recall and F1-score.

Model Performance

| Metric | Naive Bayes | Logistic Regression |

| Accuracy   | 96.05%      | 98.39% |
| Precision  | 100.00%     | 95.17% |
| Recall     | 70.47%      | 92.62% |
| F1-score   | 82.68%      | 93.88% |

How to Run the Website:
1. Clone the repository.
2. Install the required Python dependencies:
   `pip install -r requirements.txt`
3. Start the Flask application:
   `python app.py`
4. Open http://127.0.0.1:5000 in your browser.
5. Enter an SMS message and click **Analyze Message** to see whether it is classified as spam or ham.

Demonstration Video:
Download and watch the website demonstration:(demo/demo.mp4)
The video demonstrates the application running locally and shows predictions for both a spam message and a legitimate message.

Project Report:
The full project report is available in the `report` folder.
