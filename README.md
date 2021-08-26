# NeuroErgo

This method gives a differentiable model for tabular ergonomic assessment methods like Rapid Entire Body Assessment(REBA) or Rapid Upper Limb Assessment(RULA) methods. 
This is a Deep neural network model which is trained from the REBA table values as a dataset.
The proposed method is validated by comparing its results with "dREBA" method[1] which is a wellknown method in literature for modeling the REBA table.

## Structure
This is a neural network consists of 6 layer of local network (neck,trunk,leg,upper arms,lower arms, wrists) and finally with an aggregator network which returns the total REBA score based on joints degree of the local networks.
## Usage
This method can be used for human robot ergonomic optimization.

## OutPut

## Documentation

## References
<a id="1">[1]</a> 
Busch, Baptiste and Maeda, Guilherme and Mollard, Yoan and Demangeat, Marie and Lopes, Manuel (2017). 
Postural optimization for an ergonomic human-robot interaction. 
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

## TODO
