# RV_fit_covariances

This will create RV data to fit and analyze. The goal is to have a representative sample of the
planets that RV is sensitive to instead of the haphazard approach I used initally.

After generating the RV data, it will fit it with RadVel and create a covariance matrix based on the
chains.

With those covariance matrices we will attempt to create a generic covariance matrix so we can
calculate the population of orbits that could have created the RV curve quickly.
