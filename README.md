# NeuroErgo

This method gives a differentiable model for tabular ergonomic assessment methods like Rapid Entire Body Assessment(REBA) or Rapid Upper Limb Assessment(RULA) methods. 
This is a Deep neural network model which is trained from the REBA table values as a dataset.
The proposed method is validated by comparing its results with "dREBA" method[1] which is a wellknown method in literature for modeling the REBA table.

## Structure
This is a neural network consists of 6 layer of local network (neck,trunk,leg,upper arms,lower arms, wrists) and finally with an aggregator network which returns the total REBA score based on joints degree of the local networks.
## Usage
For using pre-trained models, you can loda the specified model under the data folder (in root folder). To load models, you can use the following command:

```python
from tensorflow.keras.models import load_model
model = load_model('<address to the specified model>')
```

For example, from the `main.py` function, you can load `neck` local network (for predicting local neck REBA) by the following code snippet:

```Python
neck_model = load_model('./data/neck_DNN.model')
```
Or for loading the supper model (for predicting the total REBA score) use the the following code snippet:

```Python
supper_model = load_model('./data/supper_model_DNN.model')
```

## OutPut

## Documentation

## References
<a id="1">[1]</a> 
Busch, Baptiste and Maeda, Guilherme and Mollard, Yoan and Demangeat, Marie and Lopes, Manuel (2017). 
Postural optimization for an ergonomic human-robot interaction. 
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

## TODO
