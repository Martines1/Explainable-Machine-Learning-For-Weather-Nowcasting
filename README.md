# Explainable-Machine-Learning-For-Weather-Nowcasting
This repository contains source code for Master thesis *Explainable machine learning for weather nowcasting* <br><br>
Thesis focuses on the current state of the art of explainable methods that can be used for weather nowcasting, and for proposing perturbation based post-hoc method. <br>
Methods:
- Channel-isolated Grad-CAM
- Integrated Gradients from Meteors
- Perturbation of channels
- Perturbation of unique points
<br>

Everything is covered in Python 3.12.
<br>
## Model
We use modified RainNet v1.0 to test methods. <br>
The modification involves rewriting the methods from TensorFlow to PyTorch. <br>
Official GitHub repository for RainNet : https://github.com/hydrogo/rainnet
## Installation
1. Install Python 3.12.11 <br>
2. Go to rainnet\ directory <br>
3. Open CLI and install dependencies <br>
`pip install -r requirements.txt`

## Data
We use DWD open data with *.bz2* file extension.<br>
There are two ways to obtain these data:
1. Current RY data from the following link https://opendata.dwd.de/weather/radar/radolan/ry/
2. Historical data from the following link https://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/radolan/recent/

The data must be sequential, with a 5-minute interval between each data point. There must also be exactly 4 data points for the methods to work properly. If there are 5 data points, the 5th one serves as the ground truth for comparing the prediction.<br>
Create directory in rainnet\data\ which is named after the number {c} and put downloaded data inside rainnet\data\{c}\ <br>
### Example
{c} = 0 and 4 data points are saved as:
1. \rainnet\data\0\raa01-ry_10000-2509212320-dwd---bin.bz2
2. \rainnet\data\0\raa01-ry_10000-2509212325-dwd---bin.bz2
3. \rainnet\data\0\raa01-ry_10000-2509212330-dwd---bin.bz2
4. \rainnet\data\0\raa01-ry_10000-2509212335-dwd---bin.bz2
## Usage
rainnet\ folder contains several runnable Python files
- **run.py**<br>
run basic model nowcasting with optional optical flow output and time series interpretation.
- **run_gradcam.py**<br>
run Grad-CAM method.
- **run_ig.py**<br>
run Integrated Gradient method.
- **run_pert.py**<br>
run perturbation of channels.
- **run_pert_cluster.py**<br>
run perturbation of unique points with clustering approach.
- **run_pert_window.py**<br>
run perturbation of unique points with sliding window approach.

## Output
rainnet\output contains all necessary output data for every method
