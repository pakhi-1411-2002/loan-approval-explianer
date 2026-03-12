# Explainable Loan Approval System

An interpretable machine learning system that predicts loan approvals and explains its decisions using SHAP (SHapley Additive exPlanations).

## 🎯 Project Overview

This project demonstrates how to build transparency into AI systems. Instead of just outputting "Approved" or "Rejected," it shows **why** each decision was made by breaking down feature contributions.

### Key Features

- ✅ **Real-time Predictions:** Binary classification with probability scores
- 🔍 **SHAP Explanations:** Feature attribution for every prediction
- 📊 **Visual Breakdowns:** Waterfall charts showing decision factors
- 💬 **Plain-Language Narratives:** Human-readable explanations
- 🎨 **Interactive UI:** Clean Streamlit interface

## 🚀 Live Demo

[Link to deployed app if you deploy it] OR "Run locally (see Installation below)"

## 📸 Screenshots

### Prediction Interface
[[UI Preview]](https://github.com/pakhi-1411-2002/loan-approval-explianer/blob/main/data/Screenshot%202026-03-12%20084659.png)

### SHAP Waterfall Explanation
[[SHAP Waterfall]](https://github.com/pakhi-1411-2002/loan-approval-explianer/blob/main/data/Screenshot%202026-03-12%20085219.png)

### Plain-Language Explanation
[[Explanation Panel]](https://github.com/pakhi-1411-2002/loan-approval-explianer/blob/main/data/Screenshot%202026-03-12%20085234.png)

## 🛠️ Tech Stack

- **Python 3.8+**
- **scikit-learn** - Model training & preprocessing
- **SHAP** - Model interpretability
- **Streamlit** - Web interface
- **pandas & numpy** - Data manipulation
- **matplotlib** - Visualizations

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup

1. Clone the repository
```bash
git clone https://github.com/[your-username]/loan-approval-explainer.git
cd loan-approval-explainer
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the dataset
- Place `loan_data.csv` in the `data/` folder
- https://www.kaggle.com/datasets/suryadeepthi/loan-approval-dataset

## 🎮 Usage

### Training the Model
```bash
# Preprocess data
python preprocess.py

# Train model
python train_model.py
```

### Running the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Interface

1. Enter applicant details (income, loan amount, credit score, etc.)
2. Click "Predict Loan Approval"
3. View the decision, probability, and explanation
4. Explore the SHAP waterfall chart to understand feature contributions

## 🧠 How It Works

### Architecture
```
User Input → Preprocessing → Model Prediction → SHAP Explanation → Visualization
```

### Model Pipeline

1. **Data Preprocessing**
   - One-hot encoding for categorical features
   - Standard scaling for numeric features
   - Consistent transformation between training and inference

2. **Model Training**
   - Logistic Regression classifier
   - 80/20 train-test split
   - 91% accuracy on test set

3. **Explainability**
   - SHAP LinearExplainer for feature attribution
   - Waterfall plots showing contribution breakdown
   - Template-based narrative generation

### Key Components

- `preprocess.py` - Data preprocessing and pipeline creation
- `train_model.py` - Model training and evaluation
- `app.py` - Streamlit web interface
- `src/model_utils.py` - Model loading and prediction functions
- `src/explainability.py` - SHAP computation and explanation generation
- `src/ui_components.py` - Reusable UI components

## 📊 Model Performance

- **Accuracy:** 91%
- **Precision:** 92.44%
- **Recall:** 92.25%
- **F1 Score:** 92.35%

See `models/model_metrics.txt` for detailed metrics.

## 🔍 SHAP Explanations

SHAP (SHapley Additive exPlanations) values show how each feature contributed to a specific prediction:

- **Positive values** (red bars) push toward approval
- **Negative values** (blue bars) push toward rejection
- **Magnitude** indicates strength of contribution

Example interpretation:
```
Base prediction: 0.62 (dataset average)
+ Credit Score: +0.18 (strong positive)
+ Income: +0.12 (moderate positive)
- Loan Amount: -0.08 (moderate negative)
= Final prediction: 0.84 (84% approval probability)
```

## 🎓 Learning Outcomes

Building this project taught me:

- End-to-end ML pipeline development
- Importance of preprocessing consistency
- SHAP implementation and interpretation
- Building user-facing ML applications
- Responsible AI principles and transparency

## 🔮 Future Enhancements

- [ ] Global feature importance dashboard
- [ ] Counterfactual explanations ("What would change the outcome?")
- [ ] Fairness metrics and bias detection
- [ ] Model comparison (Random Forest vs. Logistic Regression)
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] CI/CD pipeline

## 🤝 Contributing

This is a learning project, but feedback and suggestions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**⭐ If you found this project helpful, please consider giving it a star!**
```
