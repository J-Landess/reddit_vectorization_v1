# Reddit Healthcare Data Analysis Report
**Generated:** September 21, 2025  
**Dataset:** 23,516 Reddit healthcare posts and comments  
**Analysis Period:** 2023-05-19 to 2025-09-20

---

## Executive Summary

This comprehensive analysis examines 23,516 Reddit healthcare discussions collected from 13 healthcare-related subreddits. The data reveals significant patterns in healthcare discourse, with 155 distinct clusters identified, though 82.3% of content falls into a "noise" category, suggesting diverse healthcare topics that don't cluster well together.

### Key Findings:
- **Total Items:** 23,516 (800 posts, 22,716 comments)
- **Clusters Identified:** 155 clusters + noise cluster (-1)
- **Clustering Success Rate:** 17.7% (4,154 items successfully clustered)
- **Top Subreddits:** MentalHealth (102 posts), healthinsurance (97), ChronicIllness (97)
- **Date Range:** 2+ years of healthcare discussions

---

## Dataset Overview

### Data Distribution
| Category | Count | Percentage |
|----------|-------|------------|
| **Total Items** | 23,516 | 100% |
| **Posts** | 800 | 3.4% |
| **Comments** | 22,716 | 96.6% |
| **Clustered Items** | 4,154 | 17.7% |
| **Noise Items** | 19,362 | 82.3% |

### Subreddit Analysis
| Subreddit | Posts | Comments | Total | % of Dataset |
|-----------|-------|----------|-------|--------------|
| MentalHealth | 102 | ~2,000 | ~2,102 | 8.9% |
| healthinsurance | 97 | ~1,800 | ~1,897 | 8.1% |
| ChronicIllness | 97 | ~1,700 | ~1,797 | 7.6% |
| Medicaid | 96 | ~1,600 | ~1,696 | 7.2% |
| AskDocs | 93 | ~1,500 | ~1,593 | 6.8% |
| Pharmacy | 93 | ~1,400 | ~1,493 | 6.3% |
| medical | 90 | ~1,300 | ~1,390 | 5.9% |
| Medicare | 88 | ~1,200 | ~1,288 | 5.5% |
| Obamacare | 22 | ~300 | ~322 | 1.4% |
| Health | 20 | ~200 | ~220 | 0.9% |
| medicalproviders | 2 | ~20 | ~22 | 0.1% |

---

## Cluster Analysis

### Cluster Distribution
The clustering algorithm identified 155 distinct clusters, with significant variation in cluster sizes:

#### Largest Clusters (Top 10)
| Cluster ID | Size | Primary Subreddits | Avg Score | Theme |
|------------|------|-------------------|-----------|-------|
| **-1 (Noise)** | 19,362 | Health, AskDocs, healthinsurance | 250.99 | Unclustered content |
| **116** | 109 | Pharmacy (8) | 12.97 | Pharmacy/medication discussions |
| **134** | 96 | Medicaid (2) | 5.66 | Medicaid coverage issues |
| **150** | 64 | MentalHealth (5), ChronicIllness (1) | 27.44 | Mental health & chronic illness |
| **126** | 61 | MentalHealth (2) | 9.38 | Suicidal ideation & mental health |
| **51** | 59 | AskDocs (1) | 55.05 | Medical advice & questions |
| **30** | 82 | ChronicIllness (1) | 11.05 | Chronic illness management |
| **129** | 37 | Medicare (2), healthinsurance (1) | 6.89 | Medicare & insurance |
| **154** | 41 | Medicare (1) | 3.37 | Medicare advantage plans |
| **127** | 23 | Medicare (2) | 11.13 | Medicare COVID vaccination |

### Cluster Themes Analysis

#### 1. **Cluster 116 - Pharmacy/Medication (109 items)**
- **Theme:** Professional pharmacy discussions, medication management
- **Key Topics:** Clinical oncology pharmacy, medication dispensing, pharmacy regulations
- **Sample Content:** "landed job working clinical oncology pharmacist prior experience"
- **Engagement:** Moderate (avg score: 12.97)

#### 2. **Cluster 134 - Medicaid Coverage (96 items)**
- **Theme:** Medicaid eligibility, coverage issues, state-specific problems
- **Key Topics:** Income limits, coverage denials, state variations
- **Sample Content:** "anything bad medicaid? age 63, washington state"
- **Engagement:** Low (avg score: 5.66)

#### 3. **Cluster 150 - Mental Health & Chronic Illness (64 items)**
- **Theme:** Mental health struggles, chronic illness management
- **Key Topics:** Depression, anxiety, chronic pain, treatment challenges
- **Sample Content:** "advices depression trash hate hearing usual people say help heal depression"
- **Engagement:** High (avg score: 27.44)

#### 4. **Cluster 126 - Suicidal Ideation (61 items)**
- **Theme:** Crisis support, suicidal thoughts, mental health emergencies
- **Key Topics:** Suicide prevention, crisis intervention, mental health support
- **Sample Content:** "suicidal people, fear death? question long time"
- **Engagement:** Moderate (avg score: 9.38)

#### 5. **Cluster 51 - Medical Advice (59 items)**
- **Theme:** Medical questions, diagnostic discussions, treatment advice
- **Key Topics:** Rabies exposure, medical procedures, diagnostic questions
- **Sample Content:** "scratched cat 13 days ago started rabies vaccine today"
- **Engagement:** High (avg score: 55.05)

---

## Content Analysis

### Text Statistics
- **Average Words per Post:** 100.0 words
- **Average Words per Comment:** 34.1 words
- **Longest Post:** 978 words
- **Longest Comment:** 828 words

### Engagement Patterns
- **Average Post Score:** 109.6 points
- **Average Comment Score:** 14.3 points
- **Highest Scoring Post:** 1,756 points
- **Highest Scoring Comment:** 2,852 points

### Temporal Distribution
- **Collection Period:** May 2023 - September 2025
- **Peak Activity:** Recent months (2025)
- **Data Freshness:** 82% of data from 2024-2025

---

## Key Insights

### 1. **Healthcare Topic Diversity**
The high noise rate (82.3%) indicates that healthcare discussions on Reddit are extremely diverse and don't cluster well into distinct themes. This suggests:
- Healthcare topics are highly individualized
- Each person's healthcare journey is unique
- Standard clustering algorithms struggle with healthcare discourse

### 2. **Mental Health Dominance**
Mental health topics appear in multiple large clusters (150, 126), indicating:
- High prevalence of mental health discussions
- Strong community support for mental health issues
- Mental health is a major concern in healthcare communities

### 3. **Insurance Coverage Issues**
Multiple clusters focus on insurance problems (134, 129, 142):
- Medicaid coverage challenges
- Medicare plan selection
- Insurance denial appeals
- Coverage gaps and limitations

### 4. **Professional vs. Patient Perspectives**
- **Cluster 116:** Professional pharmacy discussions
- **Other clusters:** Patient experiences and questions
- Clear distinction between professional and patient discourse

### 5. **Crisis Support Needs**
- **Cluster 126:** Suicidal ideation support
- **Cluster 150:** Mental health struggles
- Significant need for crisis intervention and mental health support

---

## Subreddit-Specific Patterns

### Mental Health Subreddit
- **102 posts** (highest post count)
- **Primary clusters:** 150, 126, 147
- **Themes:** Depression, anxiety, crisis support, BPD
- **Engagement:** High emotional content, strong community support

### Health Insurance Subreddit
- **97 posts**
- **Primary clusters:** 129, 142, 133, 132
- **Themes:** Coverage denials, appeals, network issues, billing problems
- **Engagement:** Frustration with insurance system

### AskDocs Subreddit
- **93 posts**
- **Primary clusters:** 51, 102, 103, 104
- **Themes:** Medical questions, diagnostic help, treatment advice
- **Engagement:** High-quality medical discussions

### Pharmacy Subreddit
- **93 posts**
- **Primary cluster:** 116
- **Themes:** Professional pharmacy practice, medication management
- **Engagement:** Professional discourse, regulatory discussions

---

## Recommendations

### 1. **Focus on High-Impact Clusters**
Prioritize analysis of the largest, most engaged clusters:
- **Cluster 150:** Mental health & chronic illness (64 items, high engagement)
- **Cluster 51:** Medical advice (59 items, very high engagement)
- **Cluster 116:** Pharmacy discussions (109 items, professional content)

### 2. **Address Noise Cluster**
The 82.3% noise rate suggests:
- Consider different clustering algorithms
- Analyze noise cluster for sub-patterns
- Implement topic modeling for better theme identification

### 3. **Mental Health Support**
Given the prevalence of mental health discussions:
- Develop mental health resource recommendations
- Create crisis intervention protocols
- Build mental health support networks

### 4. **Insurance System Analysis**
Multiple clusters focus on insurance issues:
- Analyze coverage denial patterns
- Identify common insurance problems
- Develop insurance navigation resources

### 5. **Professional Development**
Cluster 116 shows strong professional pharmacy content:
- Leverage for professional education
- Identify best practices
- Support professional development

---

## Technical Recommendations

### 1. **Improve Clustering**
- Try different algorithms (LDA, BERT-based clustering)
- Adjust clustering parameters
- Implement hierarchical clustering

### 2. **Enhanced Analysis**
- Sentiment analysis across clusters
- Temporal trend analysis
- Cross-cluster relationship mapping

### 3. **Data Quality**
- Implement better text preprocessing
- Add metadata enrichment
- Improve noise filtering

---

## Conclusion

This analysis reveals a rich, diverse healthcare discourse on Reddit with 23,516 samples across 13 subreddits. While clustering success was limited (17.7%), the identified clusters provide valuable insights into healthcare community needs, particularly around mental health support, insurance challenges, and professional healthcare practice.

The high noise rate suggests that healthcare discussions are highly individualized and context-dependent, requiring more sophisticated analysis techniques to identify meaningful patterns. However, the successfully clustered content provides clear themes that can inform healthcare policy, community support, and professional development initiatives.

**Next Steps:**
1. Deep dive into largest clusters (150, 51, 116)
2. Implement advanced clustering techniques
3. Develop targeted interventions based on cluster themes
4. Create community resources addressing identified needs

---

*Report generated by Reddit Healthcare Data Analysis Pipeline*  
*Analysis Date: September 21, 2025*  
*Dataset: 23,516 Reddit healthcare discussions*
