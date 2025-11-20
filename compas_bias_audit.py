"""
COMPAS Recidivism Dataset Bias Audit
Using IBM AI Fairness 360 (AIF360) toolkit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.explainers import MetricTextExplainer
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_preprocess_data(filepath):
    """Load and preprocess COMPAS dataset"""
    print("Loading COMPAS dataset...")
    df = pd.read_csv(filepath)
    
    # Filter data similar to ProPublica analysis
    # Remove rows where charge date is more than 30 days from COMPAS screening
    df = df[df['c_days_from_compas'] <= 30]
    df = df[df['c_days_from_compas'] >= -30]
    
    # Filter to only include African-American and Caucasian for comparison
    df = df[df['race'].isin(['African-American', 'Caucasian'])]
    
    # Create binary labels
    # High risk = decile_score >= 5 (medium or high risk)
    df['high_risk'] = (df['decile_score'] >= 5).astype(int)
    
    # Ground truth: did they actually recidivate?
    df['recidivated'] = df['is_recid'].astype(int)
    
    # Select relevant columns
    df_clean = df[['race', 'sex', 'age', 'priors_count', 'high_risk', 'recidivated']].copy()
    
    # Remove any rows with missing values
    df_clean = df_clean.dropna()
    
    print(f"Dataset loaded: {len(df_clean)} records")
    print(f"Race distribution:\n{df_clean['race'].value_counts()}")
    print(f"\nHigh risk predictions: {df_clean['high_risk'].sum()} ({df_clean['high_risk'].mean()*100:.1f}%)")
    print(f"Actual recidivism: {df_clean['recidivated'].sum()} ({df_clean['recidivated'].mean()*100:.1f}%)")
    
    return df_clean

def create_aif360_dataset(df):
    """Convert pandas DataFrame to AIF360 BinaryLabelDataset"""
    # Encode sex as binary (0 = Female, 1 = Male)
    df_encoded = df.copy()
    df_encoded['sex_encoded'] = (df_encoded['sex'] == 'Male').astype(int)
    
    # Protected attribute: race (0 = Caucasian, 1 = African-American)
    df_encoded['race_encoded'] = (df_encoded['race'] == 'African-American').astype(int)
    
    # Prepare DataFrame with all required columns for AIF360
    # Features: sex, age, priors_count
    # Protected attribute: race_encoded
    # Labels: high_risk (predictions from COMPAS)
    aif360_df = pd.DataFrame({
        'sex': df_encoded['sex_encoded'],
        'age': df_encoded['age'],
        'priors_count': df_encoded['priors_count'],
        'race': df_encoded['race_encoded'],
        'high_risk': df_encoded['high_risk']
    })
    
    # Create dataset
    dataset = BinaryLabelDataset(
        favorable_label=0,  # Low risk is favorable
        unfavorable_label=1,  # High risk is unfavorable
        df=aif360_df,
        label_names=['high_risk'],
        protected_attribute_names=['race'],
        privileged_protected_attributes=[[0]],  # Caucasian (0) is privileged - must be list of lists
        unprivileged_protected_attributes=[[1]]  # African-American (1) is unprivileged - must be list of lists
    )
    
    return dataset, df_encoded

def calculate_bias_metrics(dataset, df_encoded):
    """Calculate various bias metrics using AIF360"""
    print("\n" + "="*60)
    print("BIAS METRICS ANALYSIS")
    print("="*60)
    
    # Create ground truth dataset for classification metrics
    # Use actual recidivism as ground truth
    y_true = df_encoded['recidivated'].values
    y_pred = df_encoded['high_risk'].values
    
    # Calculate dataset metrics (for predictions only)
    dataset_metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=[{'race': 0}],  # Caucasian
        unprivileged_groups=[{'race': 1}]  # African-American
    )
    
    explainer = MetricTextExplainer(dataset_metric)
    
    print("\n1. STATISTICAL PARITY DIFFERENCE")
    print(f"   Value: {dataset_metric.statistical_parity_difference():.4f}")
    print(f"   Interpretation: {explainer.statistical_parity_difference()}")
    
    # Create ground truth dataset (using actual recidivism as labels)
    # For ground truth: 0 = did not recidivate (favorable), 1 = recidivated (unfavorable)
    # Must use same structure as prediction dataset but with recidivated as label
    aif360_df_truth = pd.DataFrame({
        'sex': df_encoded['sex_encoded'],
        'age': df_encoded['age'],
        'priors_count': df_encoded['priors_count'],
        'race': df_encoded['race_encoded'],
        'high_risk': df_encoded['recidivated']  # Use same label name but with ground truth values
    })
    
    dataset_truth = BinaryLabelDataset(
        favorable_label=0,  # Did not recidivate is favorable
        unfavorable_label=1,  # Recidivated is unfavorable
        df=aif360_df_truth,
        label_names=['high_risk'],  # Same label name as prediction dataset
        protected_attribute_names=['race'],
        privileged_protected_attributes=[[0]],
        unprivileged_protected_attributes=[[1]]
    )
    
    # Create classification metric (predictions vs ground truth)
    # Note: dataset_truth has ground truth labels, dataset has predictions
    class_metric = ClassificationMetric(
        dataset_truth,  # Ground truth dataset (actual recidivism)
        dataset,  # Predictions dataset (COMPAS risk scores)
        unprivileged_groups=[{'race': 1}],
        privileged_groups=[{'race': 0}]
    )
    
    print("\n2. EQUAL OPPORTUNITY DIFFERENCE")
    print(f"   Value: {class_metric.equal_opportunity_difference():.4f}")
    print(f"   Interpretation: Difference in true positive rates (recall) between groups")
    
    print("\n3. AVERAGE ABSOLUTE ODDS DIFFERENCE")
    print(f"   Value: {class_metric.average_abs_odds_difference():.4f}")
    print(f"   Interpretation: Average of absolute differences in FPR and TPR between groups")
    
    # Calculate rates manually for better understanding
    caucasian_df = df_encoded[df_encoded['race'] == 'Caucasian']
    african_american_df = df_encoded[df_encoded['race'] == 'African-American']
    
    # False Positive Rate (predicted high risk but didn't recidivate)
    caucasian_fpr = ((caucasian_df['high_risk'] == 1) & (caucasian_df['recidivated'] == 0)).sum() / (caucasian_df['recidivated'] == 0).sum()
    aa_fpr = ((african_american_df['high_risk'] == 1) & (african_american_df['recidivated'] == 0)).sum() / (african_american_df['recidivated'] == 0).sum()
    
    # False Negative Rate (predicted low risk but did recidivate)
    caucasian_fnr = ((caucasian_df['high_risk'] == 0) & (caucasian_df['recidivated'] == 1)).sum() / (caucasian_df['recidivated'] == 1).sum()
    aa_fnr = ((african_american_df['high_risk'] == 0) & (african_american_df['recidivated'] == 1)).sum() / (african_american_df['recidivated'] == 1).sum()
    
    # True Positive Rate (predicted high risk and did recidivate)
    caucasian_tpr = ((caucasian_df['high_risk'] == 1) & (caucasian_df['recidivated'] == 1)).sum() / (caucasian_df['recidivated'] == 1).sum()
    aa_tpr = ((african_american_df['high_risk'] == 1) & (african_american_df['recidivated'] == 1)).sum() / (african_american_df['recidivated'] == 1).sum()
    
    # Positive Predictive Value (of those predicted high risk, how many actually recidivated)
    caucasian_ppv = ((caucasian_df['high_risk'] == 1) & (caucasian_df['recidivated'] == 1)).sum() / (caucasian_df['high_risk'] == 1).sum()
    aa_ppv = ((african_american_df['high_risk'] == 1) & (african_american_df['recidivated'] == 1)).sum() / (african_american_df['high_risk'] == 1).sum()
    
    print("\n4. FALSE POSITIVE RATE (FPR)")
    print(f"   Caucasian: {caucasian_fpr:.4f} ({caucasian_fpr*100:.2f}%)")
    print(f"   African-American: {aa_fpr:.4f} ({aa_fpr*100:.2f}%)")
    print(f"   Difference: {aa_fpr - caucasian_fpr:.4f} ({((aa_fpr/caucasian_fpr - 1)*100):.1f}% higher for African-Americans)")
    
    print("\n5. FALSE NEGATIVE RATE (FNR)")
    print(f"   Caucasian: {caucasian_fnr:.4f} ({caucasian_fnr*100:.2f}%)")
    print(f"   African-American: {aa_fnr:.4f} ({aa_fnr*100:.2f}%)")
    print(f"   Difference: {aa_fnr - caucasian_fnr:.4f}")
    
    print("\n6. TRUE POSITIVE RATE (TPR / Recall)")
    print(f"   Caucasian: {caucasian_tpr:.4f} ({caucasian_tpr*100:.2f}%)")
    print(f"   African-American: {aa_tpr:.4f} ({aa_tpr*100:.2f}%)")
    print(f"   Difference: {aa_tpr - caucasian_tpr:.4f}")
    
    print("\n7. POSITIVE PREDICTIVE VALUE (PPV / Precision)")
    print(f"   Caucasian: {caucasian_ppv:.4f} ({caucasian_ppv*100:.2f}%)")
    print(f"   African-American: {aa_ppv:.4f} ({aa_ppv*100:.2f}%)")
    print(f"   Difference: {aa_ppv - caucasian_ppv:.4f}")
    
    return {
        'statistical_parity_diff': dataset_metric.statistical_parity_difference(),
        'equal_opportunity_diff': class_metric.equal_opportunity_difference(),
        'avg_abs_odds_diff': class_metric.average_abs_odds_difference(),
        'caucasian_fpr': caucasian_fpr,
        'aa_fpr': aa_fpr,
        'caucasian_fnr': caucasian_fnr,
        'aa_fnr': aa_fnr,
        'caucasian_tpr': caucasian_tpr,
        'aa_tpr': aa_tpr,
        'caucasian_ppv': caucasian_ppv,
        'aa_ppv': aa_ppv
    }, df_encoded

def create_visualizations(df_encoded, metrics):
    """Create visualizations showing bias disparities"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. False Positive Rate by Race
    ax1 = plt.subplot(2, 3, 1)
    fpr_data = pd.DataFrame({
        'Race': ['Caucasian', 'African-American'],
        'False Positive Rate': [metrics['caucasian_fpr'], metrics['aa_fpr']]
    })
    bars = ax1.bar(fpr_data['Race'], fpr_data['False Positive Rate'], 
                   color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('False Positive Rate by Race\n(Predicted High Risk but Did Not Recidivate)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylim(0, max(fpr_data['False Positive Rate']) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n({height*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # Add difference annotation
    diff = metrics['aa_fpr'] - metrics['caucasian_fpr']
    ax1.annotate(f'Difference: {diff:.3f}\n({(diff/metrics["caucasian_fpr"]*100):.1f}% higher)',
                xy=(0.5, max(metrics['caucasian_fpr'], metrics['aa_fpr']) * 1.1),
                xytext=(0.5, max(metrics['caucasian_fpr'], metrics['aa_fpr']) * 1.15),
                ha='center', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # 2. False Negative Rate by Race
    ax2 = plt.subplot(2, 3, 2)
    fnr_data = pd.DataFrame({
        'Race': ['Caucasian', 'African-American'],
        'False Negative Rate': [metrics['caucasian_fnr'], metrics['aa_fnr']]
    })
    bars2 = ax2.bar(fnr_data['Race'], fnr_data['False Negative Rate'],
                    color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('False Negative Rate', fontsize=12, fontweight='bold')
    ax2.set_title('False Negative Rate by Race\n(Predicted Low Risk but Did Recidivate)',
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_ylim(0, max(fnr_data['False Negative Rate']) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n({height*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. True Positive Rate by Race
    ax3 = plt.subplot(2, 3, 3)
    tpr_data = pd.DataFrame({
        'Race': ['Caucasian', 'African-American'],
        'True Positive Rate': [metrics['caucasian_tpr'], metrics['aa_tpr']]
    })
    bars3 = ax3.bar(tpr_data['Race'], tpr_data['True Positive Rate'],
                    color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax3.set_title('True Positive Rate by Race\n(Correctly Predicted High Risk)',
                  fontsize=12, fontweight='bold', pad=15)
    ax3.set_ylim(0, max(tpr_data['True Positive Rate']) * 1.2)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n({height*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Positive Predictive Value by Race
    ax4 = plt.subplot(2, 3, 4)
    ppv_data = pd.DataFrame({
        'Race': ['Caucasian', 'African-American'],
        'Positive Predictive Value': [metrics['caucasian_ppv'], metrics['aa_ppv']]
    })
    bars4 = ax4.bar(ppv_data['Race'], ppv_data['Positive Predictive Value'],
                    color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Positive Predictive Value', fontsize=12, fontweight='bold')
    ax4.set_title('Positive Predictive Value by Race\n(Accuracy of High Risk Predictions)',
                  fontsize=12, fontweight='bold', pad=15)
    ax4.set_ylim(0, max(ppv_data['Positive Predictive Value']) * 1.2)
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n({height*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # 5. Risk Score Distribution by Race
    ax5 = plt.subplot(2, 3, 5)
    caucasian_scores = df_encoded[df_encoded['race'] == 'Caucasian']['high_risk']
    aa_scores = df_encoded[df_encoded['race'] == 'African-American']['high_risk']
    
    ax5.hist([caucasian_scores, aa_scores], bins=[0, 0.5, 1.5], 
             label=['Caucasian', 'African-American'],
             color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Risk Prediction', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax5.set_title('Risk Score Distribution by Race', fontsize=12, fontweight='bold', pad=15)
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['Low Risk', 'High Risk'])
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Comparison of Key Metrics
    ax6 = plt.subplot(2, 3, 6)
    metrics_comparison = pd.DataFrame({
        'Metric': ['FPR', 'FNR', 'TPR', 'PPV'],
        'Caucasian': [metrics['caucasian_fpr'], metrics['caucasian_fnr'], 
                     metrics['caucasian_tpr'], metrics['caucasian_ppv']],
        'African-American': [metrics['aa_fpr'], metrics['aa_fnr'],
                           metrics['aa_tpr'], metrics['aa_ppv']]
    })
    
    x = np.arange(len(metrics_comparison['Metric']))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, metrics_comparison['Caucasian'], width,
                    label='Caucasian', color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax6.bar(x + width/2, metrics_comparison['African-American'], width,
                    label='African-American', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax6.set_ylabel('Rate', fontsize=12, fontweight='bold')
    ax6.set_title('Key Metrics Comparison', fontsize=12, fontweight='bold', pad=15)
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics_comparison['Metric'])
    ax6.legend(fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    ax6.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('compas_bias_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualizations saved to 'compas_bias_analysis.png'")
    plt.show()
    
    # Create a second figure for additional analysis
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Actual vs Predicted Recidivism by Race
    ax1 = axes[0]
    comparison_data = []
    for race in ['Caucasian', 'African-American']:
        race_df = df_encoded[df_encoded['race'] == race]
        comparison_data.append({
            'Race': race,
            'Predicted High Risk': race_df['high_risk'].mean(),
            'Actual Recidivism': race_df['recidivated'].mean()
        })
    
    comp_df = pd.DataFrame(comparison_data)
    x = np.arange(len(comp_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, comp_df['Predicted High Risk'], width,
                    label='Predicted High Risk', color='#e67e22', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, comp_df['Actual Recidivism'], width,
                    label='Actual Recidivism', color='#27ae60', alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('Proportion', fontsize=12, fontweight='bold')
    ax1.set_title('Predicted vs Actual Recidivism by Race', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_df['Race'])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Error rates breakdown
    ax2 = axes[1]
    error_data = {
        'Caucasian': {
            'False Positives': metrics['caucasian_fpr'] * (df_encoded[df_encoded['race'] == 'Caucasian']['recidivated'] == 0).sum() / len(df_encoded[df_encoded['race'] == 'Caucasian']),
            'False Negatives': metrics['caucasian_fnr'] * (df_encoded[df_encoded['race'] == 'Caucasian']['recidivated'] == 1).sum() / len(df_encoded[df_encoded['race'] == 'Caucasian']),
            'True Positives': metrics['caucasian_tpr'] * (df_encoded[df_encoded['race'] == 'Caucasian']['recidivated'] == 1).sum() / len(df_encoded[df_encoded['race'] == 'Caucasian']),
            'True Negatives': (1 - metrics['caucasian_fpr']) * (df_encoded[df_encoded['race'] == 'Caucasian']['recidivated'] == 0).sum() / len(df_encoded[df_encoded['race'] == 'Caucasian'])
        },
        'African-American': {
            'False Positives': metrics['aa_fpr'] * (df_encoded[df_encoded['race'] == 'African-American']['recidivated'] == 0).sum() / len(df_encoded[df_encoded['race'] == 'African-American']),
            'False Negatives': metrics['aa_fnr'] * (df_encoded[df_encoded['race'] == 'African-American']['recidivated'] == 1).sum() / len(df_encoded[df_encoded['race'] == 'African-American']),
            'True Positives': metrics['aa_tpr'] * (df_encoded[df_encoded['race'] == 'African-American']['recidivated'] == 1).sum() / len(df_encoded[df_encoded['race'] == 'African-American']),
            'True Negatives': (1 - metrics['aa_fpr']) * (df_encoded[df_encoded['race'] == 'African-American']['recidivated'] == 0).sum() / len(df_encoded[df_encoded['race'] == 'African-American'])
        }
    }
    
    error_df = pd.DataFrame(error_data).T
    error_df.plot(kind='bar', stacked=True, ax=ax2, color=['#e74c3c', '#f39c12', '#27ae60', '#3498db'],
                  alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Proportion of Group', fontsize=12, fontweight='bold')
    ax2.set_title('Error Type Breakdown by Race', fontsize=12, fontweight='bold', pad=15)
    ax2.legend(title='Outcome Type', fontsize=9, title_fontsize=10)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('compas_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("Detailed analysis saved to 'compas_detailed_analysis.png'")
    plt.show()

def generate_report(metrics, df_encoded):
    """Generate a 300-word report summarizing findings"""
    report = f"""COMPAS RECIDIVISM DATASET BIAS AUDIT REPORT

EXECUTIVE SUMMARY
This audit analyzed racial bias in the COMPAS recidivism risk assessment tool using IBM's AI Fairness 360 toolkit. The analysis examined disparities between African-American and Caucasian defendants in risk score predictions and actual recidivism outcomes.

KEY FINDINGS

1. False Positive Rate Disparity
The most significant finding is a substantial disparity in false positive rates (FPR) - cases where defendants were predicted as high risk but did not recidivate. African-American defendants exhibit a false positive rate of {metrics['aa_fpr']:.1%}, compared to {metrics['caucasian_fpr']:.1%} for Caucasian defendants. This represents a {((metrics['aa_fpr']/metrics['caucasian_fpr'] - 1)*100):.1f}% higher rate for African-Americans.

2. Statistical Parity Difference
The statistical parity difference of {metrics['statistical_parity_diff']:.3f} indicates unequal treatment across racial groups. A value closer to zero would indicate fairness, suggesting bias.

3. Prediction Accuracy
While the positive predictive value shows {metrics['aa_ppv']:.1%} for African-Americans versus {metrics['caucasian_ppv']:.1%} for Caucasians, the false positive rate disparity remains the primary concern. This suggests the algorithm predicts high risk more frequently for African-Americans.

4. Dataset Characteristics
The analysis included {len(df_encoded[df_encoded['race'] == 'Caucasian']):,} Caucasian and {len(df_encoded[df_encoded['race'] == 'African-American']):,} African-American defendants, with recidivism rates of {df_encoded[df_encoded['race'] == 'Caucasian']['recidivated'].mean():.1%} and {df_encoded[df_encoded['race'] == 'African-American']['recidivated'].mean():.1%} respectively.

REMEDIATION STEPS

1. Algorithmic Interventions: Implement fairness-aware machine learning techniques such as reweighing, adversarial debiasing, or equalized odds post-processing.

2. Threshold Adjustment: Apply race-specific thresholds for risk classification to ensure equal false positive rates across groups.

3. Feature Audit: Review and remove features that may serve as proxies for race, ensuring the model focuses on legitimate risk factors.

4. Regular Monitoring: Establish ongoing bias monitoring protocols using metrics like false positive rate and statistical parity.

5. Transparency: Increase transparency in risk score calculation and provide explanations for predictions.

CONCLUSION
The audit reveals significant racial bias in the COMPAS risk assessment tool, particularly in false positive rates. These findings align with previous research and highlight the need for bias mitigation strategies in criminal justice risk assessment systems."""
    
    # Save report to file
    with open('bias_audit_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("REPORT GENERATED")
    print("="*60)
    print(report)
    print(f"\nReport saved to 'bias_audit_report.txt'")
    print(f"Report word count: {len(report.split())} words")

def main():
    """Main execution function"""
    print("="*60)
    print("COMPAS RECIDIVISM DATASET BIAS AUDIT")
    print("Using IBM AI Fairness 360 Toolkit")
    print("="*60)
    
    # Load and preprocess data
    filepath = 'COMPAS-Recidivism-Dataset/compas-scores.csv'
    df_clean = load_and_preprocess_data(filepath)
    
    # Create AIF360 dataset
    dataset, df_encoded = create_aif360_dataset(df_clean)
    
    # Calculate bias metrics
    metrics, df_encoded = calculate_bias_metrics(dataset, df_encoded)
    
    # Create visualizations
    create_visualizations(df_encoded, metrics)
    
    # Generate report
    generate_report(metrics, df_encoded)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - compas_bias_analysis.png (main visualizations)")
    print("  - compas_detailed_analysis.png (detailed breakdown)")
    print("  - bias_audit_report.txt (300-word report)")

if __name__ == "__main__":
    main()

