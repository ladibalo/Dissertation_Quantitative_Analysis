# SDOH Completeness Analysis - Dissertation Code

## Overview

This repository contains Python scripts for the complete quantitative analysis of the dissertation: **"The variability and completeness of social determinants of health (SDOH) data collection in the endocrinology clinics at UI Health, Chicago treating patients with type 2 diabetes."**

**Author:** Sulaimon Balogun, PhD Candidate  
**Institution:** Department of Biomedical and Health Information Sciences, UI Health Chicago  
**Study Population:** 10,609 Type 2 diabetes patients  
**Study Period:** September 2020 - September 2025  

## 📁 Repository Contents

### Core Analysis Scripts

1. **`dissertation_final_analysis.py`**
   - **Purpose:** Comprehensive end-to-end dissertation quantitative analysis
   - **Scope:** Complete analysis pipeline covering all research objectives
   - **Key Features:**
     - Patient-year window SDOH completeness methodology
     - Encounter-level (1.08%) and patient-level (3.7%) analysis
     - Domain-specific performance evaluation (13 SDOH categories)
     - Demographic and temporal trend analysis
     - Statistical testing and comprehensive visualizations

2. **`multivariable_regression_analysis.py`**
   - **Purpose:** Focused regression modeling with comprehensive diagnostics
   - **Scope:** Patient and provider-level predictive modeling
   - **Key Features:**
     - Patient-level logistic regression (10,609 patients, 391 events)
     - Provider-level linear regression (24 providers, R² = 0.74)
     - Model assumption testing and validation
     - Statistical significance assessment and effect size reporting

## 📊 Key Analysis Results

### Overall Performance
- **Encounter-Level Completeness:** 1.08% (2,203/204,470 encounters)
- **Patient-Level Completeness:** 3.7% (391/10,609 patients)
- **3.4-fold increase** from encounter to patient level due to patient-year window

### Domain Performance Rankings
| Domain | Coverage | Classification | High Risk |
|--------|----------|----------------|-----------|
| Tobacco Use | 97.2% | Excellence | 12.1% |
| Food Insecurity | 53.2% | Moderate | 9.3% |
| Financial Resource Strain | 35.2% | Low | 6.8% |
| Physical Activity | 26.3% | Crisis | 35.5% |

### Statistical Models

**Patient-Level Logistic Regression:**
- Female Sex: OR = 1.47 (95% CI: 1.23-1.76), p < 0.001
- Age per year: OR = 1.01 (95% CI: 1.002-1.018), p = 0.008
- Healthcare Utilization: OR = 1.10 (95% CI: 1.03-1.18), p = 0.008

**Provider-Level Linear Regression:**
- Screening Intensity: β = 0.68, p = 0.002 (strongest predictor)
- Domain Breadth: β = 0.45, p = 0.028
- Non-Physician Specialists: β = 0.52, p = 0.013
- Fellow Status: β = -0.48, p = 0.023

## 🗃️ Data Requirements

### Expected Input Files
```
req_2026_00831_demo.csv        # Patient demographics (10,764 records)
req_2026_00831_visits.csv      # Clinical encounters (1,759,655 records)
req_2026_00831_sdoh.csv        # SDOH screening data (580,116 records)
req_2026_00831_diagnoses.csv   # Diagnostic codes (43,574 records)
req_2026_00831_labs.csv        # Laboratory results (666,713 records)
req_2026_00831_vitals.csv      # Vital signs (6,722,909 records)
```

### Data Structure Requirements
- **Patient ID:** Consistent identifier across all datasets (`pat_id`)
- **Date Formats:** ISO format (YYYY-MM-DD) for all date fields
- **SDOH Domains:** 13 standardized categories as defined in study protocol
- **Study Period:** September 1, 2020 - September 30, 2025

## 🚀 Usage Instructions

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```

### Running the Analysis

**Option 1: Complete Analysis**
```bash
python dissertation_final_analysis.py
```
*Runs the full dissertation analysis pipeline (recommended)*

**Option 2: Focused Regression Analysis**
```bash
python multivariable_regression_analysis.py
```
*Runs detailed regression modeling with diagnostics*

### Expected Output Files
- `outputs/dissertation_final_results_summary.md` - Comprehensive findings report
- `outputs/dissertation_comprehensive_analysis.png` - Multi-panel visualization
- `outputs/multivariable_regression_analysis.png` - Regression results charts

## 🎯 Clinical Applications

### Immediate Quality Improvement Targets
1. **Address Physical Activity Crisis** (highest risk, lowest screening)
2. **Eliminate Gender Disparities** (1.2 percentage point gap)
3. **Insurance Equity Initiative** (2.6 percentage point Medicare vs Self-Pay gap)

### Proven Success Model
- **Tobacco Use Screening:** Demonstrates >96% systematic coverage is achievable
- **Systematic Protocols:** Provider screening intensity explains 74% of performance variation
- **Replicable Excellence:** Apply tobacco model to all 13 SDOH domains

## 📈 Technical Specifications

### Core Methodology
- **Patient-Year Window:** SDOH screenings credited to all encounters in same calendar year
- **Completeness Definition:** Documentation of all 13 SDOH domains within patient-year
- **Statistical Approach:** Mixed descriptive, comparative, and predictive modeling

### Quality Assurance
- **Data Quality:** 99.97% complete after automated preprocessing
- **Statistical Rigor:** All assumptions tested, confidence intervals reported
- **Reproducibility:** Complete methodology documentation and error handling

## 📚 Academic Standards

This code provides the dissertation quantitative analysis framework:
- ✅ **Methodological Rigor:** Comprehensive statistical analysis methodology
- ✅ **Statistical Validity:** Full model diagnostics and assumption testing
- ✅ **Reproducibility:** Complete code documentation with comprehensive comments
- ✅ **Clinical Relevance:** Actionable findings for healthcare quality improvement
- ✅ **Publication Ready:** Academic-standard tables, figures, and statistical reporting

## 🔗 Repository Structure

```
├── dissertation_final_analysis.py          # Main comprehensive analysis
├── multivariable_regression_analysis.py    # Focused regression modeling
├── README.md                               # This documentation
└── outputs/                               # Generated analysis results
    ├── *.md                              # Markdown reports
    └── *.png                             # Visualization files
```

## 📄 Citation

If using this code for academic purposes, please cite:

```
Balogun, S. (2026). The variability and completeness of social determinants 
of health (SDOH) data collection in the endocrinology clinics at UI Health, 
Chicago treating patients with type 2 diabetes. Department of Biomedical 
and Health Information Sciences, University of Illinois Chicago.
```

---

**Contact:** For questions regarding methodology or implementation, please refer to the comprehensive inline documentation within each script or contact the dissertation committee at UI Health Chicago.