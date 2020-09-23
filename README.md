# SocialNavEval

## Usage
This tool is used to analyze various metrics for social navigation paths. It is designed as flexible for new formats/data types. \




## Metrics Supported
All metrics are reported with mean, std_dev, min, max
* CollisionsPerAgent 
* ExtraTimeToGoal
  * Extra Time to Goal 
  * Path Efficiency (ratio of straight path to agent path)
* AgentClosestProximity
* PathIrregularity
* AverageAgentPositionDerivatives
  * Speed
  * Acceleration
  * Jerk
* SocialScore
* CongestionScore (Percentage of time spent below threshold speed)
* PopulationDensity
* SimilarityScores
  * Frechet Distance
  * Dynamic Time Warping
 * PersonalSpace (sampled agent's mean Voronoi region area)
 * ScenarioArea (Minimum bbox bounding a scenarios agents)
    
## Dependencies: 
Clone and run to install getpy `setup.py`: https://github.com/atom-moyer/getpy \
`pip install similaritymeasures`\
`pip install matplotlib`\
`pip install xmltodict`\
`pip install pandas`\
`pip install scipy`
