# Executive Summary: Salifort Motors Employee Retention Analysis

## Project Overview
This analysis utilized machine learning techniques to predict employee turnover at Salifort Motors and identify key factors affecting retention. The Random Forest model achieved 98% accuracy and 0.97 F1 score, providing reliable predictions to support HR decision-making.

## Key Findings

### Primary Factors Affecting Employee Turnover
1. **Satisfaction Level** (Importance: 0.21)
   - Employees who left had a significantly lower average satisfaction level (0.44) compared to those who stayed (0.67)
   - Satisfaction is the single most important predictor of turnover

2. **Number of Projects** (Importance: 0.17)
   - Employees with too many projects (6+) showed higher turnover
   - Optimal range appears to be 3-4 projects

3. **Time Spent at Company** (Importance: 0.15)
   - Employees who had been at the company for 3-4 years without promotion showed higher turnover
   - Both too short (<1 year) and extended periods (5+ years) without advancement correlated with departures

4. **Average Monthly Hours** (Importance: 0.14)
   - Employees working excessive hours (>250 hours/month) were more likely to leave
   - Employees who left worked longer hours on average (277 vs. 199 hours/month)

5. **Last Evaluation Score** (Importance: 0.11)
   - Both very high (>0.9) and very low (<0.5) evaluation scores correlated with turnover
   - Pattern suggests burnout in high performers and disengagement in low performers

### Department and Salary Insights
- Technical departments (IT, technical) showed higher turnover rates
- Lower salary levels correlated with higher turnover across most departments
- Management roles had the lowest turnover regardless of salary level

## Model Performance
- **Random Forest Classifier**: 98% accuracy, 0.97 F1 score
- Outperformed Logistic Regression (83% accuracy), Decision Tree (95% accuracy), and XGBoost (96% accuracy)
- Model demonstrates strong precision (0.96) and recall (0.95), minimizing both false positives and false negatives

## Recommendations

### Immediate Actions (0-3 months)
1. **Implement Regular Satisfaction Surveys**
   - Conduct monthly pulse surveys to monitor employee satisfaction
   - Create dedicated feedback channels for concerns

2. **Review Workload Distribution**
   - Audit project assignments across departments
   - Establish guidelines to limit project assignments to 3-4 per employee
   - Implement controls on overtime hours

3. **Enhance Recognition Programs**
   - Develop a structured recognition system for contributions
   - Ensure recognition is timely, specific, and meaningful

### Short-term Initiatives (3-6 months)
1. **Develop Career Development Programs**
   - Create clear career paths with milestone achievements
   - Implement quarterly career development discussions
   - Focus on employees in the 3-4 year tenure range

2. **Improve Work-Life Balance**
   - Establish policies to prevent excessive working hours
   - Monitor and limit overtime consistently
   - Provide flexible work arrangements where possible

3. **Review Compensation Structure**
   - Conduct market analysis of compensation by department
   - Address salary disparities in high-turnover departments

### Long-term Strategies (6-12 months)
1. **Implement Comprehensive Retention Programs**
   - Develop department-specific retention strategies
   - Create tailored approaches for high-risk employee segments

2. **Develop Succession Planning**
   - Identify key positions and potential successors
   - Create development plans for high-potential employees

3. **Establish Mentorship Programs**
   - Pair experienced employees with newer staff
   - Create cross-departmental knowledge-sharing opportunities

## Implementation Timeline
The recommendations follow a staged approach to balance immediate impact with long-term structural improvements. Initial focus is on addressing satisfaction and workload issues, followed by career development and compensation adjustments, culminating in comprehensive retention programs.

## Conclusion
The analysis provides clear evidence that employee satisfaction, project workload, working hours, and career development opportunities are the primary drivers of turnover at Salifort Motors. By addressing these factors through the recommended initiatives, the company can significantly improve retention rates and reduce the costs associated with employee turnover. 