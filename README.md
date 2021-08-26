# NeuroErgo

This method gives a differentiable model for tabular ergonomic assessment methods like Rapid Entire Body Assessment(REBA) or Rapid Upper Limb Assessment(RULA) methods. 
This is a Deep neural network model which is trained from the REBA table values as a dataset.
The proposed method is validated by comparing its results with "dREBA" method[1] which is a wellknown method in literature for modeling the REBA table.

## Structure
This is a neural network consists of 6 layer of local network (neck,trunk,leg,upper arms,lower arms, wrists) and finally with an aggregator network which returns the total REBA score based on joints degree of the local networks.

## Preparing Models
For training each model, i.e., local networks and aggregator network, there is a function with the name format of `<name_of_body_part>_training_model` inside `./main.py` (under the root folder). For example, for the neck part, this function is dubbed `neck_training_model` or for the total aggregated network `total_reba_from_partial_training_model`. Also, for the super model, combining aggregator and local networks, you can find `super_model_train` function. The only need for creating models is calling corresponding models in the `./main.py` file, like the following:

```Python
neck_training_model()
trunk_training_model()
leg_training_model()
upper_arm_training_model()
lower_arm_training_model()
total_reba_from_partial_training_model()

generate_super_model_training_data()
super_model_train()
```

Note that for training the super model, we are using 4 billion generated data. 
As generating this data volume is time-consuming, and keeping it in the memory is not viable,
first, we must call `generate_super_model_training_data()` by chunking data into the small files (near million data samples in each file).
Then, by reading them, train the super model. 

Also, if you need to modfiy the topology or data ranges for each netwrok, you can find a function with the name format of `<body_part_name>_learning_model` and `<body_part_name>_learning_ranges`, respectively, in the `./main.py` file. 
For the supper model, combining the networks is done by `create_super_model` function.

## Usage
For using pre-trained models, you can loda the specified model under the data folder (in root folder). To load models, you can use the following command:

```python
from tensorflow.keras.models import load_model
model = load_model('<address to the specified model>')
```

For example, from the `./main.py` function (under root folder), you can load `neck` local network (for predicting local neck REBA) by the following code snippet:

```Python
neck_model = load_model('./data/neck_DNN.model')
```
Or for loading the supper model (for predicting the total REBA score) use the the following code snippet:

```Python
supper_model = load_model('./data/supper_model_DNN.model')
```

After loading the model, you can do the approximation by the following call, for example the super model prediction:

```Python
num_of_data = 2
data = {
    'neck_model_input': np.zeros(shape=(num_of_data, 3)),
    'trunk_model_input': np.zeros(shape=(num_of_data, 3)),
    'leg_model_input': np.zeros(shape=(num_of_data, 1)), 
    'upper_arm_model_input': np.zeros(shape=(num_of_data, 6)), 
    'lower_arm_model_input': np.zeros(shape=(num_of_data, 2)), 
    'wrist_model_input': np.zeros(shape=(num_of_data, 6))
}
data['neck_model_input'][:, :] = [[10, 10, 10], [20, 20, 20]]
data['trunk_model_input'][:, :] = [[10, 10, 10], [20, 20, 20]]
data['leg_model_input'][:, :] = [[10], [20]]
data['upper_arm_model_input'][:, :] = [[10, 10, 10, 10, 10, 10], [20, 20, 20, 20, 20, 20]]
data['lower_arm_model_input'][:, :] = [[10, 10], [20, 20]]
data['wrist_model_input'][:, :] = [[10, 10, 10, 10, 10, 10], [20, 20, 20, 20, 20, 20]]

pred = super_model.predict(data)
pred = list(chain(*pred)) # list of two REBA scores for two input body joints 
```



## OutPut

## Documentation

## References
<a id="1">[1]</a> 
Busch, Baptiste and Maeda, Guilherme and Mollard, Yoan and Demangeat, Marie and Lopes, Manuel (2017). 
Postural optimization for an ergonomic human-robot interaction. 
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

## TODO
