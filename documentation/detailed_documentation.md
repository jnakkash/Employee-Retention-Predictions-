# Detailed Documentation: Salifort Motors Employee Retention Predictive Model

## Project Description
This project implements a machine learning predictive model for Salifort Motors to identify employees at risk of leaving the company. By analyzing various employee attributes and behaviors, the model provides insights on potential turnover risks and enables HR to develop proactive retention strategies.

## Data Understanding

### Dataset Overview
The dataset consists of 14,999 employee records with 10 features:
- **satisfaction_level**: Employee satisfaction score (0-1)
- **last_evaluation**: Score of last performance evaluation (0-1)
- **number_project**: Number of projects assigned
- **average_monthly_hours**: Average monthly working hours
- **time_spend_company**: Years employed at the company
- **Work_accident**: Whether the employee had a workplace accident (binary)
- **left**: Whether the employee left the company (binary, target variable)
- **promotion_last_5years**: Whether the employee was promoted in the last 5 years (binary)
- **Department**: Department the employee belongs to (categorical)
- **salary**: Salary level (low, medium, high)

### Data Quality Assessment
The following data quality checks were performed:
- Missing values detection and handling
- Outlier identification
- Distribution analysis of each feature
- Correlations between features
- Class balance of the target variable

## Methodology

### Data Preprocessing

#### Data Cleaning
- Fixed inconsistent column names (e.g., "average_montly_hours" to "average_monthly_hours")
- Standardized data types
- Handled outliers using IQR method and capping

#### Feature Engineering
- **Categorical Encoding**: One-hot encoding for 'Department' and 'salary' features
- **Interaction Features**:
  - workload: hours per project
  - work_intensity: ratio of hours worked to standard 160 monthly hours
  - evaluation_per_project: ratio of evaluation score to number of projects
  - satisfaction_evaluation_ratio: ratio of satisfaction to evaluation score
  - overworked: binary indicator for employees with >200 hours and >4 projects

#### Data Transformation
- StandardScaler applied to numeric features
- Both scaled and unscaled versions preserved for interpretability

### Modeling Approach

#### Model Selection
Four different classification algorithms were implemented and compared:
1. **Logistic Regression**: A linear model for binary classification
2. **Random Forest**: An ensemble learning method using multiple decision trees
3. **XGBoost**: A gradient boosting algorithm
4. **Decision Tree**: A simple, interpretable tree-based model

#### Hyperparameter Tuning
- GridSearchCV with 5-fold cross-validation
- Optimized for F1 score
- Hyperparameters tuned for each model:
  - Logistic Regression: C, penalty, solver, class_weight
  - Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf, class_weight
  - XGBoost: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, scale_pos_weight
  - Decision Tree: max_depth, min_samples_split, min_samples_leaf, criterion, class_weight

#### Model Training
- Train-test split: 70% training, 30% testing
- Class imbalance handled through class weights
- Cross-validation to prevent overfitting

### Evaluation Methodology

#### Performance Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Accuracy of positive predictions (minimizes false positives)
- **Recall**: Ability to find all positive instances (minimizes false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

#### Feature Importance Analysis
- Built-in feature importance for tree-based models
- Coefficients for logistic regression
- Permutation importance for more reliable feature importance measurement

## Implementation Details

### Exploratory Data Analysis Script (`exploratory_analysis.py`)
- Visualizes distributions of all features
- Explores relationships between features and target
- Generates correlation matrices
- Identifies patterns in employee turnover

### Data Cleaning Script (`data_cleaning.py`)
- Performs data quality checks
- Handles inconsistencies and outliers
- Engineers new features
- Applies transformations
- Outputs processed datasets

### Model Training Script (`model_training.py`)
- Prepares data for modeling
- Trains multiple models with hyperparameter tuning
- Evaluates model performance
- Visualizes model comparisons
- Saves trained models

### Model Evaluation Script (`model_evaluation.py`)
- Performs in-depth evaluation of best model
- Analyzes feature importance and effects
- Visualizes model insights
- Generates executive summary with recommendations

## Error Handling and Debugging

### Data Validation Checks
- Input data validation at each step
- Checks for missing features
- Validation of categorical values
- Output format verification

### Exception Handling
- Try-except blocks for file operations
- Graceful handling of missing data
- Validation of model outputs
- Clear error messaging

## Performance Results
*Note: Exact performance metrics will be available after running the models on the dataset.*

## Ethical Considerations

### Privacy
- The dataset contains no personally identifiable information
- All analysis is done at an aggregate level

### Fairness
- Model evaluation includes checking for biases across departments and salary levels
- Multiple models are compared to ensure fair prediction outcomes

### Transparency
- Feature importance analysis provides transparency in decision-making
- Model selection prioritizes interpretability alongside performance

## Recommendations and Business Impact

### Key Insights
- Identification of top factors influencing employee turnover
- Patterns in employee behavior preceding departure
- Department-specific retention challenges

### Actionable Recommendations
- Targeted interventions for high-risk employees
- Policy adjustments based on key turnover factors
- Structural changes to improve retention

### Implementation Strategy
- Development of monitoring dashboard
- Integration with HR systems
- Regular model updates with new data

## Future Enhancements

### Model Improvements
- Testing of additional algorithms (e.g., neural networks)
- Ensemble methods combining multiple models
- Time-series analysis for temporal patterns

### Feature Expansion
- Inclusion of additional HR metrics when available
- Text analysis of performance reviews
- Integration of team dynamics data

### Production Deployment
- API development for real-time predictions
- Integration with HR information systems
- Automated reporting and alerting system 