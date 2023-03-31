# High-Resolution Working Layouts and Time Series for Renewable Energy Generation in Europe: A Data-Driven Approach for Accurate Fore- and Nowcasting

### Oliver Grothe, Fabian Kächele, Mira Watermeyer

The stability and manageability of power systems with a growing share of renewable energies depend on accurate forecasts and feed-in information. This study provides synthetic wind and solar power generation time series for approximately 1,500 European transmission nodes in hourly resolution from 2019 to 2022, along with data-driven layouts of renewable generator allocations.
To create these time series and layouts, we develop weather-to-energy conversions using high-resolution weather data. Based on the conversions and elastic-net optimisation, the layouts, which we refer to as working layouts, represent a theoretical allocation of generators within each country that produces the current (or alternatively any historical) observed energy output characteristics.
The layouts may be employed for accurate, high spatial resolution energy production forecasts and nowcasts. This work provides the necessary code to update and adapt layouts and time series for use in custom applications.

### Keywords:
Renewable Energy, Layout, Renewable Energy Capacity, Renewable Generation, Weather-to-Energy Conversion, Solar power, Wind power, Electricity Grid
 
### Links: 
tba

### Explanations: 
The code reproduces the benchmarks from the paper. It can easily be updated to generate results for a specific time horizon and location and tailor the layouts and feed-ins to specific needs.

In folder "Data and configuration files" we provide data sheets for the used wind turbines and solar module. Additionally, an Excel-file with the coordinates of all network nodes is provided. 

In folder "Scripts" users can find all implementations needed to reproduce the data set. It consists of the following functions: 

        - overall model: 
            "main_code.py"
            
        - data import: 
            "weather_data()", 
            "feedin_data()"
            
        - calculation of relative signals for onshore wind, offshore wind and 
            PV generation with weather-to-energy conversions: 
            "windcalculation()", 
            "solarcalculation()"
            
        - weather cell combination: 
            "weather_cell_combination()"
            
        - layout estimation: 
            "layout_estimation()"
            
The functions are included in the main file: "main_code.py", which has to be run to reproduce the data. 

### Citing IntEG

The model published in this repository is free: you can access, modify and share it under the terms of the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. This model is shared in the hope that it will be useful for further research on topics of risk-aversion, investments, flexibility and uncertainty in ectricity markets but without any warranty of merchantability or fitness for a particular purpose. 

If you use the model or its components for your research, we would appreciate it if you
would cite us as follows:
```
This paper is in review. The reference to the working paper version is as follows:

 O. Grothe, F. Kächele, and M. Watermeyer (2023), High-Resolution Working Layouts and Time Series for Renewable Energy Generation 
 in Europe: A Data-Driven Approach for Accurate Fore- and Nowcasting, Working paper
```
