# Watts Up: Forecasting and Optimizing Massachusetts’ EV Charging Network

**Author:** Aarav Singh Chahar  
**Degree:** Master of Science in Business Analytics, University of Massachusetts Boston

---

## 1. Project Overview

This project provides a comprehensive, data-driven assessment of Massachusetts’ public electric vehicle (EV) charging infrastructure. It combines time-series forecasting, spatial analysis, clustering, and machine learning to answer a central planning question:

> **How can Massachusetts optimize its EV charging infrastructure rollout—both by deploying new stations and upgrading existing ones—for maximum impact?**

Using the U.S. Department of Energy’s Alternative Fuels Data Center (AFDC) dataset, the analysis:

- Quantifies the current charging landscape (capacity, provider mix, charger types).
- Forecasts the monthly number of new charging stations using ARIMA and SARIMA models.
- Simulates a +10% EV adoption scenario and its impact on infrastructure needs.
- Segments stations into meaningful business archetypes using K-Means clustering.
- Identifies geographic hotspots, underserved regions, and future growth corridors.
- Models the charging network as a graph to find critical hubs and bridges.
- Prioritizes candidates for DC-fast charger upgrades with a Random Forest classifier.

The goal is to provide a set of **clear, actionable recommendations** for policymakers, utilities, and charging-network operators.

---

## 2. EV Adoption and Policy Context

Globally, EV adoption has accelerated sharply. In the last year, nearly **14 million** electric cars were sold worldwide—a **35% increase** over 2022—with EVs comprising **18%** of all new-car sales. The International Energy Agency (IEA) projects that EVs could reach **40%** of global auto sales by 2030, with further gains into the mid-2030s.

In the United States, EV registrations are growing but still account for only **10%** of global EV registrations. Within Massachusetts:

- All-electric registrations rose from **49,400** at the end of 2022 to **68,842** in 2023.
- As of January 2024, there were **66,000** zero-emission vehicles (ZEVs) on the road.
- The state’s goal is **200,000** ZEVs by **2025**, a target that requires a sharp acceleration in adoption.

On the infrastructure side:

- Massachusetts currently has approximately **6,436** public charging ports.
- The Clean Energy & Climate Plan calls for **15,000** ports by **2025** and **75,000** by **2030**.

In simple terms, the charging network must **more than double in the near term** and expand several-fold by 2030 in order to support the state’s EV and climate targets. This project is designed to provide data-driven guidance for that expansion.  [oai_citation:1‡Watts up findings.docx](sediment://file_00000000403871f5b7dbc30634c8230b)

---

## 3. Data Ingestion and Preparation

### 3.1 Source and Initial Filtering

The core dataset is the U.S. Department of Energy’s Alternative Fuels Data Center record of public EV charging stations in Massachusetts. It originally contains:

- **4,042 rows** and **75 columns**, each row representing a charging station.

To focus on infrastructure that is actually in service:

- The data is filtered to stations marked as **“Existing”**, reducing the sample to **3,859** active locations.

### 3.2 Feature Selection

Many of the 75 original attributes are either sparsely populated or not relevant to the analytical objectives. The data is trimmed to **10 essential fields** that capture station identity, location, capacity, and connectivity:

- **Station identification:**  
  - Station Name  
  - City

- **Geospatial attributes:**  
  - Latitude  
  - Longitude

- **Temporal attribute:**  
  - Open Date (used to construct a monthly time series)

- **Capacity and technology mix:**  
  - EV Level1 EVSE Num  
  - EV Level2 EVSE Num  
  - EV DC Fast Count

- **Network and connectors:**  
  - EV Network  
  - EV Connector Types

All other fields with >90% missingness or minimal analytical relevance are removed, yielding a compact, interpretable dataset suitable for modeling.

### 3.3 Feature Engineering

Several derived variables are constructed to summarize each station’s capabilities and configuration:

- **`total_ports`**: sum of Level-1, Level-2, and DC-fast ports.  
- **`fast_ratio`**: DC-fast ports divided by `total_ports`.  
- **`card_count`**: number of distinct payment methods accepted.  
- **`connector_type_count`**: count of distinct connector types per station.  
- **`install_month`**: installation date converted to a monthly timestamp, producing a continuous monthly series from **March 2009** through **May 2025**.

The resulting dataset combines a clean, well-defined feature set with rich temporal and spatial information. It underpins all subsequent analyses, including time-series forecasting, spatial gap mapping, clustering, and prioritization of fast-charger upgrades.  [oai_citation:2‡Watts up findings.docx](sediment://file_00000000403871f5b7dbc30634c8230b)

---

## 4. Exploratory Analysis

### 4.1 Station Size and Charger Mix

A set of numeric summaries is computed for each station’s Level-1, Level-2, and DC-fast port counts, along with the engineered `card_count` and `connector_type_count` features. This clarifies both typical station sizes and outliers.

Key findings:

- **Level-2 prevalence:**  
  - Stations in Massachusetts are heavily oriented toward Level-2 charging.  
  - The average site offers **2.14** Level-2 ports.  
  - Approximately **75%** of locations have **exactly two Level-2** ports.  
  - This suggests a network designed primarily for slower, neighborhood-style charging, rather than rapid throughput.

- **DC-fast charger shortfall:**  
  - The mean number of DC-fast ports per station is **0.31**.  
  - Roughly **three-quarters of sites have no DC-fast option at all**.  
  - In practical terms, about four out of five chargers will not support fast charging, leaving a significant gap for drivers who need rapid turnaround.

### 4.2 Network Concentration

The distribution of stations across EV networks is highly skewed:

- **ChargePoint** operates **2,979** of the **3,859** active stations, accounting for about **77%** of the network.
- **Non-networked** stations account for around **240** locations (~6%).
- The remaining eight networks collectively share the final ~17%.

This concentration highlights both potential advantages (standardization, easier integration) and risks (dependence on a single provider), and it informs partnership and diversification strategies.  [oai_citation:3‡Watts up findings.docx](sediment://file_00000000403871f5b7dbc30634c8230b)

### 4.3 Installation Trends and Seasonality

A time-series view of `install_month` reveals distinct trend and seasonal components:

- **Trend:**  
  - From 2009 to 2018, installs remain nearly flat, with very few new stations added in most months.  
  - Starting in **mid-2019**, installations increase sharply as policies, incentives, and network investments take effect.  
  - The **12-month rolling average** climbs from fewer than **5 stations per month** in 2018 to over **50 per month** by 2021, then stabilizes in the **50–70 installs per month** range through 2024.

- **Seasonality:**  
  - A strong annual pattern emerges.  
  - Each summer, installs climb to roughly **30 stations above trend**, while winter months often dip about **10 stations below trend**.  
  - These cycles likely reflect construction schedules, permitting windows, and higher project activity in warmer months.

- **Residuals and outliers:**  
  - Most monthly deviations from the trend-plus-seasonal model are within ±20 stations, suggesting the main structure is well captured.  
  - A notable outlier is early **2021**, when installs spike above **400 stations** in a short window, likely due to a significant funding tranche or accelerated program rollout.

These observations motivate the use of models that explicitly account for both trend and seasonality.

---

## 5. Time-Series Modeling: ARIMA and SARIMA

### 5.1 Introducing SARIMA for Seasonal Forecasting

To capture both the underlying growth in installations and the clear annual cycle, the project uses:

- A non-seasonal **ARIMA(1,1,1)** model as a benchmark.
- A **Seasonal ARIMA (SARIMA)** model, **SARIMA(1,1,1) × (1,1,1,12)**, which introduces seasonal autoregressive and moving-average components at a 12-month lag.

SARIMA is particularly appropriate given the summer peaks and winter lulls identified in exploratory analysis. It produces forecasts that respect both the trend and recurring seasonal swings, enabling month-by-month targets aligned with construction and permitting cycles.

### 5.2 SARIMA Results

The SARIMA model is trained on monthly station counts from **January 2009 to May 2024**. Performance metrics are:

- **In-sample performance:**  
  - RMSE = **40.46**  
  - MAE  = **13.66**  
  These indicate typical monthly errors of 14–40 stations on the training period.

- **Hold-out performance (June 2024–May 2025):**  
  - RMSE = **25.65**  
  - MAE  = **19.46**  
  The model maintains strong performance when predicting unseen data, confirming that it is capturing the core patterns, including seasonality.  [oai_citation:4‡Watts up findings.docx](sediment://file_00000000403871f5b7dbc30634c8230b)

The SARIMA forecasts show:

- Predicted summer months with approximately **75–95** new stations.
- Predicted winter months with approximately **55–65** new stations.
- A 12-month projection that maintains a **±30-station** seasonal amplitude into mid-2026.

### 5.3 Forecast Tables: Test Set and Projections

To evaluate the model quantitatively, the project compares:

- **Actual vs. forecasted** installation counts for the 12-month hold-out period (June 2024–May 2025), alongside the **95% confidence intervals** (`lower_CI` and `upper_CI`) around each forecast.

Illustrative rows:

- 2024-06-01: actual = 47, forecast ≈ 64.05, CI range from roughly –22.38 to 150.49.  
- 2024-07-01: actual = 105, forecast ≈ 49.12, CI range from roughly –37.35 to 135.59.

In addition, the model forecasts **June 2025–May 2026**, providing projected monthly counts and associated confidence intervals. These projections offer concrete numerical targets for planning, while also conveying the uncertainty range around each month.

### 5.4 ARIMA vs. SARIMA Benchmark Comparison

The benchmark comparison over the same 12-month hold-out period yields:

- **ARIMA(1,1,1):**  
  - RMSE = **24.10**  
  - MAE  = **17.22**

- **SARIMA(1,1,1) × (1,1,1,12):**  
  - RMSE = **25.65**  
  - MAE  = **19.46**

Interpretation:

- ARIMA offers **slightly lower numerical error**, deviating by roughly 17–24 stations per month.
- SARIMA sacrifices a small amount of accuracy (≈2 extra stations of error) in exchange for **explicit seasonal modeling**, which yields more realistic forecasts over the course of a full year and better alignment with observed summer/winter patterns.

### 5.5 ARIMA Baseline and Planning Implications

The ARIMA model is trained on 15 years of monthly installation counts, capturing trend and short-term fluctuations:

- **In-sample performance:**  
  - MAE = **11.32**  
  - RMSE = **39.47**

- **Hold-out performance:**  
  - MAE = **17.22**  
  - RMSE = **24.10**

The ARIMA baseline forecasts hover around **55–60 new stations per month**, offering a steady reference trajectory.

**Business interpretation:**

- Use **SARIMA** for **operational planning**, because it aligns with seasonal construction and permitting cycles.
- Use **ARIMA** as a **quick baseline** to verify that forecasts remain within reasonable error bounds.

---

## 6. Scenario Simulation: +10% EV Registration Surge

To understand how a faster-than-expected rise in EV adoption would pressure the network, the project simulates a **+10% increase** in EV registrations over the next year.

Assuming baseline installs of **55–60 stations per month**, the scenario implies:

- A need for **approximately 5–6 additional stations per month**.
- A cumulative increase of about **65–70 additional stations** over the year.

Examples:

- June 2025: the baseline forecast of roughly 57 new stations would rise to about 63.
- July 2025–May 2026: each month would require an additional 5–6 stations.

**Business actions:**

1. **Early permitting:** begin permit applications in advance to avoid summer bottlenecks.  
2. **Crew and hardware planning:** ramp staffing and place equipment orders early to ensure availability.  
3. **Grid readiness in hotspots:** coordinate with utilities in Boston, Worcester, and Lowell to ensure the grid can handle higher demand and additional fast-charging capacity.  [oai_citation:5‡Watts up findings.docx](sediment://file_00000000403871f5b7dbc30634c8230b)

This proactive stance reduces the risk of undersupply and service delays when EV adoption accelerates.

---

## 7. Station Segmentation: Business-Relevant Archetypes

To prioritize upgrades and tailor deployment strategies, the project uses **K-Means clustering** to segment stations into groups based on:

- Total ports (`total_ports`)
- DC-fast share (`fast_ratio`)
- Connector-type diversity
- Payment option diversity (`card_count`)

Features are standardized, and **Principal Component Analysis (PCA)** is used to reduce dimensionality to two components for clearer visualization. Silhouette analysis is employed to select the number of clusters, after which K-Means is fitted.

### 7.1 Cluster Archetypes

The analysis identifies **five clear archetypes**:

1. **Neighborhood Level-2 Sites**  
   - Typically ~2 Level-2 ports, no fast chargers.  
   - Ideal for residential complexes and local retail parking where vehicles dwell for extended periods.

2. **Compact Fast-Charge Kiosks**  
   - One or two ports with more than half of capacity devoted to DC-fast.  
   - Suited to quick top-ups at roadside stops or busy urban locations.

3. **High-Speed Hubs**  
   - Approximately nine ports, predominantly DC-fast.  
   - Designed for highway corridors and fleet depots where high throughput and rapid turnaround are essential.

4. **Mega Level-2 Parks**  
   - Large facilities with around twenty ports, nearly all Level-2.  
   - Common in campuses, malls, or office garages where dwell times are long and fast charging was not a priority at build-out.

5. **Mid-Tier Mixed Sites**  
   - Around eight ports with one or two DC-fast chargers.  
   - Balanced use cases, serving both slow and occasional fast-charge needs.

Mapping these clusters back to geography reveals where each station type is prevalent and how they align with existing travel patterns and demand.

### 7.2 Tier Interpretation and Strategic Recommendations

In the tier plot (stations as colored points, cluster centers as black markers), each cluster exhibits a distinct profile:

- **Tier 0 (Neighborhood L2):** ~2 Level-2 ports, no fast.  
- **Tier 1 (Compact Fast Kiosks):** 2–4 ports, with 50–75% DC-fast.  
- **Tier 2 (High-Speed Hubs):** ~10 ports, nearly all fast.  
- **Tier 3 (Mega L2 Parks):** 20+ ports, almost entirely Level-2.  
- **Tier 4 (Mid-Tier Mixed):** ~8 ports, small but nonzero fast-charger share.

**Strategic implications:**

- Expand **Tier 2** high-speed hubs along major commuter and freight routes.  
- Retrofit **Tier 3** bulk Level-2 parks with DC-fast chargers to capture missed high-value charging opportunities.  
- Deploy more **Tier 1** compact fast kiosks at key pinch points and high-traffic locations.  

These moves generate a more balanced and flexible charging network that better matches drivers’ real needs.

---

## 8. Spatial Hotspots and Capacity Gaps

### 8.1 Geographic Distribution of Charging Hubs

Spatial density plots show where chargers are clustered:

- High-density hubs emerge in **Boston**, **Worcester**, **Springfield**, and **New Bedford**.
- Much of **western and central Massachusetts** appears as low-density “charging deserts”.

**Business implications:**

- High-density zones represent **proven demand**, suitable for further medium and fast-charger deployment.  
- Underserved regions indicate potential for strategic new builds, both to support adoption and to meet equity or policy goals.  
- Station type should match regional demand profiles (for instance, fast kiosks near busy commuter routes and mixed sites in emerging suburban markets).  [oai_citation:6‡Watts up findings.docx](sediment://file_00000000403871f5b7dbc30634c8230b)

### 8.2 Projected Capacity Needs Under a +10% Adoption Scenario

A spatial overlay projects **additional station needs** under the +10% EV growth scenario at the ZIP-code level:

- **Largest bubbles** (highest additional station requirements) appear around **Boston, Worcester, and Springfield**.  
- **Smaller bubbles** in rural ZIPs suggest that existing networks can absorb increased demand with minimal expansion.

**Deployment guidance:**

- Pre-stage construction crews in and around Boston, Worcester, and Springfield.  
- Place early orders for hardware in these ZIP codes to avoid supply and installation bottlenecks.  
- Prioritize fast-track permitting in areas where modeled capacity needs spike.

---

## 9. Charging Network Connectivity

The charging network is modeled as a **graph**, where:

- Each station is a node.
- Edges connect stations located within a five-mile radius.

### 9.1 Hubs: Degree Centrality

By measuring **degree centrality**, the analysis identifies stations with many nearby neighbors—the network hubs. These are concentrated in:

- **Downtown Boston**, including sites on or near Berkeley Street.

These hubs act as focal points for regional access and are prime candidates for capacity expansion and the introduction of additional fast-charging capabilities or new services.

### 9.2 Bridges: Betweenness Centrality

**Betweenness centrality** highlights stations that sit on many shortest paths between other stations—key “bridges” connecting different parts of the network. Notable bridge locations include:

- **Shirley**  
- **Ayer**  
- **Boxborough**  
- **Worcester**

Although these stations may not be the largest in terms of port count, they play a critical role in maintaining connectivity between suburban and urban clusters. Ensuring high uptime, good signage, and adequate capacity at these bridges yields outsized benefits for overall network reliability.

---

## 10. Fast-Charger Upgrade Prioritization (Random Forest)

To identify stations that are strong candidates for DC-fast charger upgrades, the project trains a **Random Forest classifier** on station attributes, including:

- Total port count  
- Network affiliation (e.g., Tesla, EVgo, ChargePoint)  
- Graph-based measures such as betweenness centrality

### 10.1 Model Performance and Drivers

- The model achieves a **near-perfect ROC AUC (~1.00)**, indicating almost complete separation between stations with and without DC-fast chargers.
- The most important predictor is **`total_ports`** (importance ≈ 0.64), followed by:
  - Network brands such as **Tesla** and **EVgo**.
  - Network centrality metrics, especially **betweenness centrality**.

This implies that **larger stations** and those already integrated into major fast-charging networks are the best upgrade candidates.

### 10.2 Geographic Prioritization

When stations are ranked by their model-predicted probability of supporting or benefiting from DC-fast chargers:

- The **Boston metro area** features prominently among the top-ranked sites.  
- Additional high-priority pockets appear in **Worcester** and **Springfield**.

**Planning application:**

- Use this ranked list as a **shortlist for fast-charger upgrades**, directing permitting, equipment orders, and grid upgrades toward these high-impact locations first.

---

## 11. Key Business Insights and Recommendations

Drawing together the forecasting, clustering, spatial, network, and classification analyses, the project supports several key recommendations:

1. **Prioritize fast-charger upgrades at large, highly connected stations.**  
   Focus on stations with **more than eight ports** and high network centrality, particularly those already on **Tesla** or **EVgo** networks. These locations combine scale and connectivity, maximizing the impact of additional DC-fast capacity.

2. **Expand into growth corridors identified by the +10% scenario.**  
   Concentrate near-term expansion in **Springfield** and **Worcester**, which emerge as critical growth corridors in the scenario maps. Begin **pre-permitting** and **pre-ordering equipment** now to stay ahead of demand.

3. **Tailor incentives by station cluster.**  
   - **Small “neighborhood” sites (Clusters 0/1):**  
     Provide grants or subsidies to support the installation of **their first DC-fast port**, unlocking new use cases and improving local service.  
   - **Large “hub” sites (e.g., Tier 3 / Cluster 3):**  
     Use **volume discounts** and structured incentive programs to encourage multi-port fast-charger upgrades, leveraging economies of scale.

4. **Reinforce bridge stations to improve resilience.**  
   Maintain high uptime, clear signage, and, where needed, additional capacity at bridge stations such as **Shirley**, **Ayer**, **Boxborough**, and **Worcester**, as they knit together different parts of the statewide network.

5. **Align infrastructure expansion with seasonal patterns.**  
   Use SARIMA-based forecasts to schedule construction, permitting, and workforce allocation in line with seasonal peaks and troughs. Plan aggressive build-out activities for warmer-weather periods and use slower months for design, permitting, and grid coordination.

Together, these recommendations provide a structured roadmap for scaling Massachusetts’ EV charging network in a way that is **data-driven, spatially targeted, and operationally realistic**.

---

## 12. Repository Structure

A typical repository structure for this project is:

```text
watts-up-ev-analytics/
├── watts_up_ev_analytics.py       # Main analysis script
├── README.md                      # This documentation
├── requirements.txt               # Python dependencies
├── visuals/                       # Key plots and tables exported as images
└── maps/                          # HTML maps (Folium, etc.)