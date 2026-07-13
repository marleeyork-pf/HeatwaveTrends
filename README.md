<!-- <h2>  HeatwaveTrends README  (HTML) <h2> -->

<h1>HeatwaveTrends</h1>
<p></strong>Analyzing trends in novel heatwave definitions and applying recurrent neural network to understand legacy effects</strong></p

<hr>

<h2>TLDR</h2>
<p>Existing heatwave definitions often fail to capture the full range of heat stress experienced by ecosystems; therefore, I developed a comprehensive heatwave classification framework and evaluated how different forms of heat stress influence gross primary production (GPP), ecosystem respiration (Reco), and net ecosystem exchange (NEE) across diverse ecosystems in the western United States. We integrated long-term (1994–2024) carbon flux observations from 54 FLUXNET sites with historical climate data to identify distinct heatwave types and quantify ecosystem responses during and following heatwave events.
</p>

<h2>Data Visualization</h2>
<p>
  <img src="figures/US-Ton_2012_mean_temperature_heatwave_detection.png" width="400" alt="Identification of Heatwaves">
  <img src="figures/neeresponseheatmap.png" width="400" alt="NEE Response Heatmap">
  <img src="figures/hw_seasonality.png" width="400" alt="Seasonal Trends in Heatwave Types">
  <img src="figures/moisture_IGBP.png" width="400" alt="Random Forest Variable Importances and Interactions">
  <img src="figures/hw_dominance_map.png" width="400" alt="Spatial Dominance of Heatwave Types Across Sites">
</p>

<h2>Findings</h2>
<ul>
  <li><strong>The heat stress experienced by ecosystems is more complex than traditional heatwaves capture.</strong> Of the seven heatwave categories identified, the traditional daytime-only heatwaves (defined solely by maximum temperature exceedances) accounted for 20.9% of all heatwave days (Fig. 3). In contrast, 62.3% of heatwave days involved exceedances in daily mean temperature (overall, day-intensified, night-intensified, and triad heatwaves), while 44.7% involved elevated nighttime temperature (nighttime, night-intensified, day-night spike, and triad heatwaves).</li>
  <li><strong>Heat stress complexity influenced the direction of ecosystem carbon responses during heatwaves. </strong>The proportion of positive and negative NEE responses differed significantly among the seven categories (p = 0.007) during a heatwave event (Table 1). Single-metric heatwaves (daytime, nighttime, and overall) consistently exhibited the lowest proportion of positive NEE responses. When heatwave categories were grouped by heat stress complexity, multi-dimensional heat waves exhibited a significantly greater proportion of positive NEE responses than single-metric heatwaves (p = 0.012).</li>
  <li><strong>Heatwave categories exhibited distinct recovery trajectories in ecosystem carbon responses during the legacy period.</strong> Differences in the proportion of positive and negative NEE responses remained present through 15 days following heatwave events (p < 0.05). Heatwave categories such as overall and nighttime dropped to and maintained relatively low proportions of positive NEE responses throughout the legacy period, while day and day-night spike heatwaves held relatively high proportions of positive NEE responses throughout the legacy period (Fig. 4). Alternatively, triad, day-intensified, and night-intensified heatwaves showed elevated proportions of positive NEE responses during the heatwave that gradually declined to near or below 50% throughout the legacy period. </li>
</ul>

<h2>Highlights</h2>
<ul>
  <li><strong>Detailed QA/QC</strong>: adjusted satellite data based on in-situ tower station data</li>
  <li><strong>Satellite Data Bias Adjustment</strong>: adjusted satellite data based on in-situ tower station data</li>
  <li><strong>Heatwave Algorithm</strong>: crafted algorithm to define various types of heatwaves based on historical temperatures</strong></li>
  <li><strong>Analytics</strong>: performed statistical testing and trend analysis to determine heatwave impacts</li>
</ul>


<h2>Skills</h2>
<ul>
  <li><strong>Languages</strong>: Python</li>
  <li><strong>Libraries</strong>: TensorFlow, pandas, matplotlib</li>
  <li><strong>Compute</strong>: HPC cluster for large-scale processing</li>
  <li><strong>Data Source</strong>: <a href="https://prism.oregonstate.edu/">PRISM Data</a></li>
</ul>

<h2>Workflow Overview</h2>
<pre>
carbonflux/
├── figures/                # Data Visualizations from analysis
├── preprocessing/          # Functions and scripts for cleaning and adjusting data
├── heatwave_definition/    # Key functions and scripts for defining and analyzing heatwaves
└── README.md
</pre>

<h2>Contact</h2>

<p>
  For questions or collaboration: <strong>marleeyork2025@gmail.com</strong><br>
<p>
