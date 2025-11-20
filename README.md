# AI Ethics Course Project

This repository contains the complete work for the AI Ethics course assignment, including case study analyses, practical bias audit implementation, and ethical reflections.

## 📁 Project Structure

```
AI-Ethics/
│
├── Answers.pdf                          # Case study analyses and reflections (Parts 1, 2, 4)
├── README.md                             # This file - project overview and documentation
├── requirements.txt                      # Python package dependencies
│
├── compas_bias_audit.py                  # Main Python script for COMPAS bias audit (Part 3)
├── bias_audit_report.txt                 # Generated 300-word report summarizing findings
│
├── compas_bias_analysis.png             # Main visualizations (6 charts)
├── compas_detailed_analysis.png          # Additional detailed analysis charts
│
└── COMPAS-Recidivism-Dataset/           # Dataset directory
    ├── compas-scores.csv                 # Main dataset file used in analysis
    ├── compas-scores-raw.csv
    ├── compas-scores-two-years.csv
    ├── compas-scores-two-years-violent.csv
    ├── cox-parsed.csv
    └── cox-violent-parsed.csv
```

## 📋 Assignment Components

### Part 1: Case Study Analysis (in Answers.pdf)
- Analysis of ethical issues in AI systems
- Case study evaluations and reflections

### Part 2: Case Study Analysis (in Answers.pdf)
- Additional case study analyses
- Ethical considerations and reflections

### Part 3: Practical Audit (25%)
- **COMPAS Recidivism Dataset Bias Audit**
- Implementation using IBM AI Fairness 360 (AIF360) toolkit
- Racial bias analysis in risk scores
- Visualizations showing disparities in false positive rates
- 300-word report summarizing findings and remediation steps

**Deliverables:**
- ✅ `compas_bias_audit.py` - Complete bias analysis code
- ✅ `compas_bias_analysis.png` - Main visualizations
- ✅ `compas_detailed_analysis.png` - Detailed breakdown
- ✅ `bias_audit_report.txt` - 300-word report

### Part 4: Ethical Reflection (5%) (in Answers.pdf)
- Personal project reflection (COMPAS bias audit)
- Framework for ensuring ethical AI principles in future projects
- Practical implementation strategies

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. Clone or download this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

**Note**: AIF360 may require additional system dependencies. If you encounter issues, refer to the [AIF360 installation guide](https://github.com/Trusted-AI/AIF360).

**Expected Warnings**: When running the script, you may see warnings about missing `fairlearn` and `inFairness` modules. These are optional dependencies and can be safely ignored - the core analysis functionality works without them.

### Running the COMPAS Bias Audit

Execute the main analysis script:
```bash
python compas_bias_audit.py
```

## 📊 Part 3: COMPAS Bias Audit Details

### What It Does

The `compas_bias_audit.py` script performs a comprehensive bias analysis of the COMPAS recidivism risk assessment tool:

1. **Data Loading & Preprocessing**
   - Loads COMPAS dataset from CSV
   - Filters to cases within 30 days of COMPAS screening (following ProPublica methodology)
   - Focuses on African-American and Caucasian defendants
   - Creates binary risk classifications (high risk = decile_score >= 5)

2. **Bias Metrics Calculation**
   - Uses IBM AI Fairness 360 toolkit to calculate:
     - Statistical Parity Difference
     - Equal Opportunity Difference
     - Average Absolute Odds Difference
   - Calculates detailed metrics:
     - False Positive Rate (FPR) by race
     - False Negative Rate (FNR) by race
     - True Positive Rate (TPR) by race
     - Positive Predictive Value (PPV) by race

3. **Visualization Generation**
   - Creates comprehensive charts showing:
     - False Positive Rate disparities (key finding)
     - False Negative Rate comparisons
     - True Positive Rate comparisons
     - Positive Predictive Value comparisons
     - Risk score distributions
     - Error type breakdowns

4. **Report Generation**
   - Generates a 300-word report summarizing:
     - Key findings
     - Bias metrics
     - Remediation steps

### Output Files

After running the script, you'll get:

1. **Console Output**: Detailed bias metrics printed to console
2. **compas_bias_analysis.png**: Main visualization dashboard (6 charts)
3. **compas_detailed_analysis.png**: Additional detailed analysis (2 charts)
4. **bias_audit_report.txt**: 300-word written report

### Key Findings

The analysis reveals significant racial bias in the COMPAS risk assessment tool:

- **False Positive Rate Disparity**: African-American defendants have a false positive rate of 47.1%, compared to 24.2% for Caucasian defendants - a **94.5% higher rate** for African-Americans
- **Statistical Parity Difference**: -0.252 (indicating unequal treatment across racial groups)
- **Equal Opportunity Difference**: -0.229 (difference in true positive rates)
- **Average Absolute Odds Difference**: 0.221 (overall fairness metric)

These findings align with previous research (ProPublica, 2016) highlighting bias in criminal justice risk assessment systems.

## 📈 Key Metrics Analyzed

- **False Positive Rate (FPR)**: Rate of incorrectly predicted high-risk cases
- **False Negative Rate (FNR)**: Rate of incorrectly predicted low-risk cases
- **True Positive Rate (TPR)**: Rate of correctly predicted high-risk cases
- **Statistical Parity Difference**: Difference in positive prediction rates between groups
- **Equal Opportunity Difference**: Difference in true positive rates between groups
- **Average Absolute Odds Difference**: Overall fairness metric

## 📚 Dataset Information

The analysis uses the COMPAS scores dataset, focusing on:
- African-American and Caucasian defendants
- Risk scores (decile_score >= 5 = high risk)
- Actual recidivism outcomes (is_recid)
- Data filtered to cases where COMPAS screening occurred within 30 days of arrest (following ProPublica's methodology)

## 🔧 Technical Details

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- aif360 >= 0.5.0
- scikit-learn >= 1.2.0

### Code Structure

The `compas_bias_audit.py` script contains:
- `load_and_preprocess_data()`: Data loading and filtering
- `create_aif360_dataset()`: Converts data to AIF360 format
- `calculate_bias_metrics()`: Computes all bias metrics
- `create_visualizations()`: Generates all charts
- `generate_report()`: Creates the 300-word report
- `main()`: Orchestrates the entire analysis

## 📄 Documentation

- **Answers.pdf**: Contains Parts 1, 2, and 4 (case study analyses and ethical reflection)
- **bias_audit_report.txt**: Generated report for Part 3
- **README.md**: This file - project documentation

## 🎯 Assignment Deliverables Summary

| Component | Status | File(s) |
|-----------|--------|---------|
| Part 1: Case Study Analysis | ✅ Complete | Answers.pdf |
| Part 2: Case Study Analysis | ✅ Complete | Answers.pdf |
| Part 3: Practical Audit | ✅ Complete | compas_bias_audit.py, visualizations, report |
| Part 4: Ethical Reflection | ✅ Complete | Answers.pdf |

## 📝 Notes

- All code is well-documented with comments explaining each step
- The analysis follows ProPublica's methodology for data filtering
- Visualizations are saved as high-resolution PNG files (300 DPI)
- The report is automatically generated with actual metrics from the analysis

## 🔗 References

- ProPublica (2016). "Machine Bias" - Investigation into COMPAS risk assessment tool
- IBM AI Fairness 360 Toolkit: https://github.com/Trusted-AI/AIF360
- COMPAS Dataset: Available in the `COMPAS-Recidivism-Dataset/` directory

## 👥 Team

**Team Members:**
- Mary Wairimu
- Fred Kaloki
- Kelvin Karani
- Odii Chinenye Gift
- Rivaldo Ouma

This is a collaborative project for the AI Ethics course assignment.

---

**Note**: This project demonstrates practical application of AI ethics principles through systematic bias auditing, visualization, and comprehensive reporting.
