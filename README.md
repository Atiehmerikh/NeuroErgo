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
For the super model, combining the networks is done by `create_super_model` function.

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
Or for loading the super model (for predicting the total REBA score) use the the following code snippet:

```Python
super_model = load_model('./data/super_model_DNN.model')
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

## Comparison
For comparing the reuslts with the dREBA, we need to implemeent the method first. 
As the dREBA method is analytical, we have implemented it with MATLAB. 
You can find the corresponding files in `./dREBA/matlab/` folder. 
For training and testing, we need to generate data. 
For this matter, you can run `./dREBA/main.py`. 
This run generates required data for training and testing in `./dREBA/matlab/data/input/` folder, called `M.csv` and `N.csv` for training, and 
`M_test.csv` and `N_test.csv`, respectively. `M` files are the sampls of body joints in each row, and `N` files are the REBA scores for each sample in the row of `M` files.

Afterwards, running `./dREBA/matlab/optimizer_tobecontinued.m` will output test results on the random generated data in `./dREBA/matlab/dat/output/` folder. 

Finally, to have test errors on the super model, you can run `./main.py` to get the results on `./data` folder with the names of `neuro_errors.csv` and `neuro_estimations.csv` that are the test error, and the approximated REBA score on the generated data, respectively.  


## Optimization

There is a chance for utilizing the learning models in a task optimization. The required forward kinematics for human body 
has been implemented in `./human_forward_kinematic.py` and obejective function for this optimiztion task is minimizing the sum of risk (REBA score) for the human worker and forward kinematics of the human body and a target that can be the target position for an instrument in the factory floor. You can find this function in `./main.py`, dubbed `objective_function`.

For the optimization, we can use the [`localsolver` library](https://www.localsolver.com/) which comprises a balck-box optimization. You can find the example in the following:

```Python
# first load the super model
super_model = load_model('./data/super_model_DNN.model')

# allowd body joint ranges (domain-oriented)
qss = [[-60,30], [-54,54], [-60,60],\
      [-30,60], [-40, 40], [-35, 35],\
      [0,60],\
      [-20,45], [-20, 0, 20, 45], [-2,0], [-2,0], [0, 30], [0, 30],\
      [0,100], [0, 100],\
      [-53,15], [-53,15], [-40, 30], [-40, 30], [-90, 90], [-90, 90]]

with localsolver.LocalSolver() as ls:
    model = ls.get_model()

    for i, qs in enumerate(qss):
        minimum = min(qs)
        maximum = max(qs)
        globals()['x%s' % i] = eval(f'model.float({minimum},{maximum})')
    f = model.create_double_blackbox_function(objective_function)
    call = model.call()
    call.add_operand(f)

    for i in range(len(qss)):
        eval(f'call.add_operand(x{i})')

    model.minimize(call)
    model.close()

    ls.get_param().set_time_limit(50)
    ls.solve()
    sol = ls.get_solution()
    for i in range(len(qss)):
        eval('print("x{} = {}".format('+ str(i) + ',sol.get_value(x' + str(i) +')))')

    # the ouput of the solver is optimized body joint degrees that minimized risk,
    # besides the distance of human forward kinematic and the target
    print("obj = {}".format(sol.get_value(call)))
```




## References
<a id="1">[1]</a> 
Busch, Baptiste and Maeda, Guilherme and Mollard, Yoan and Demangeat, Marie and Lopes, Manuel (2017). 
Postural optimization for an ergonomic human-robot interaction. 
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),
