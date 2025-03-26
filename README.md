# Employee Retention Predictive Model - Salifort Motors

## Project Overview
This project implements a machine learning predictive model to identify employees at risk of leaving Salifort Motors, enabling proactive retention strategies and reducing hiring costs. The Random 

Your goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.

If we can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

The analysis identifies key factors affecting employee turnover including satisfaction levels, number of projects, working hours, and time spent at the company. These insights form the basis for recommended interventions to improve employee retention rates.

## Key Findings
- **Satisfaction level** is the strongest predictor of employee turnover
- Employees with **high number of projects** and **long working hours** show increased turnover risk
- Employees who stayed **3-4 years** with no promotion show higher turnover rates
- **Department** and **salary level** have significant impact on retention

## Dashboard
The interactive dashboard provides a comprehensive view of the analysis results and can be accessed below:
https://jnakkash.github.io/Employee_Retention_Prediction_Model

Forest model achieves 98% accuracy in predicting employee turnover, providing HR teams with actionable insights to improve retention.

### Dashboard Features
- **Executive Summary**: Concise overview of key findings and recommendations
- **Model Comparison**: Performance metrics of all tested models with explanation of why Random Forest was selected
- **Feature Importance**: Visual representation of factors influencing employee turnover
- **Key Findings**: Interactive visualizations with dropdown menu to explore different aspects of employee data
- **Recommendations**: Actionable strategies organized by implementation timeline

## Dataset
The dataset contains information from 14,999 employees with the following features:
- `satisfaction_level` (0-1)
- `last_evaluation` (0-1)
- `number_project`
- `average_montly_hours`
- `time_spend_company` (years)
- `Work_accident` (binary)
- `left` (binary, target variable)
- `promotion_last_5years` (binary)
- `Department` (categorical)
- `salary` (categorical)

## Project Structure
```
project-root/
├── data/                         # Data directory
│   ├── raw/                      # Raw dataset
│   └── processed/                # Cleaned and processed data
│
├── documentation/                # Project documentation
│   └── executive_summary.md      # Summary of findings and recommendations
│
├── models/                       # Directory for saved trained models (created during training)
│
├── results/                      # Analysis outputs
│   ├── analyzed_data/            # EDA visualizations and findings
│   ├── model_comparison/         # Model evaluation metrics and charts
│   └── dashboard/                # Dashboard files
│
├── exploratory_analysis.py       # EDA script
├── interactive_dashboard.html    # Interactive dashboard file
├── interactive_dashboard.py      # Interactive dashboard generation script
├── model_evaluation.py           # Model evaluation script
├── model_training.py             # Model training script
├── README.md                     # Project documentation
└── run_full_analysis.py          # Full pipeline execution script
```

## Scripts Overview

### `run_full_analysis.py`
Orchestrates the complete analysis pipeline:
- Runs exploratory data analysis
- Trains machine learning models
- Evaluates model performance
- Generates visualizations and insights
- Creates the interactive dashboard

### `exploratory_analysis.py`
Performs initial data exploration:
- Loads and examines raw data
- Cleans and preprocesses the dataset
- Creates visualizations of feature distributions
- Analyzes relationships between variables
- Generates insights about employee characteristics
- Outputs visualizations to results/analyzed_data/

### `model_training.py`
Trains and optimizes machine learning models:
- Preprocesses data for modeling
- Implements multiple classification algorithms
- Performs hyperparameter tuning
- Evaluates models using cross-validation
- Selects the best performing model (Random Forest)
- Saves trained models to models/ directory

### `model_evaluation.py`
Evaluates model performance and generates insights:
- Assesses model accuracy on test data
- Calculates performance metrics (F1 score, precision, recall)
- Identifies key features driving employee turnover
- Generates visualizations of model performance
- Creates an executive summary with actionable recommendations
- Outputs visualizations to results/model_comparison/

### `interactive_dashboard.py`
Creates an interactive HTML dashboard:
- Compiles results from all analysis phases
- Generates visualizations for key findings
- Implements interactive elements with dropdown menus
- Formats the executive summary into readable sections
- Compares performance across multiple models
- Provides explanations of why Random Forest was selected
- Creates both comprehensive and simplified visualizations for business users

## Setup and Execution

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost

### Running the Project

1. **Full Analysis Pipeline**:
   ```
   python run_full_analysis.py
   ```
   This runs all analysis components in sequence.

2. **Individual Components**:
   ```
   python exploratory_analysis.py
   python model_training.py
   python model_evaluation.py
   python interactive_dashboard.py
   ```

3. **View Dashboard**:
   Open `interactive_dashboard.html` in any web browser to explore the interactive dashboard.

## Models Implemented
- **Random Forest** (best performance, selected as final model)
- Logistic Regression
- Decision Tree
- XGBoost

## Evaluation Metrics
- **Accuracy**: 98%
- **F1 Score**: 0.97
- **Precision**: 0.96
- **Recall**: 0.95

## Recommendations
Based on the analysis, we recommend the following strategies to improve employee retention:

### Immediate Actions (0-3 months)
- Implement regular satisfaction surveys and feedback mechanisms
- Review workload distribution and project assignments
- Enhance recognition and reward programs

### Short-term Initiatives (3-6 months)
- Develop career development programs
- Improve work-life balance policies
- Enhance training and development opportunities

### Long-term Strategies (6-12 months)
- Implement comprehensive retention programs
- Develop succession planning
- Create mentorship programs
