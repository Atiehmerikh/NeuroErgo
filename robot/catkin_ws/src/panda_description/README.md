# Panda description package
This package is the first package you need to do simulations with the Panda robot. 

## Visualize in Rviz
The Panda robot can be [visualized in Rviz](http://gazebosim.org/tutorials/?tut=ros_urdf#ViewinRviz) by 
```bash
roslaunch panda_description panda_rviz.launch 
```
To visualize a robot in Rviz, a urdf with the kinematics data and meshes are required to make the visualization as similar as possible to reality. 
The Panda's robot urdf and meshes can be found on [Franka Emika's github account](https://github.com/frankaemika/franka_ros/tree/kinetic-devel/franka_description).

## Visualize in Gazebo
To be able to launch the Panda robot in Gazebo the [urdf folder](ros_ws/src/panda_description/urdf) in the panda_description package has been adapted such that the robot dynamics are included. 
1. panda_arm_hand.urdf: rigidly fix the base to the Gazebo world
2. hand.xacro: add inertial values
3. panda_arm.xacro: add inertial values + add joint damping
4. panda.gazebo.xacro: new file with gazebo specifications
5. panda_arm_hand.urdf.xacro: include panda.gazebo.xacro

## Control the Panda robot in Gazebo
To be able to control the Panda robot in Gazebo, the [urdf folder](ros_ws/src/panda_description/urdf) in the panda_description package has been adapted such that actuators are linked to joints. 
1. panda_arm.xacro: add transmission elements
2. hand.xacro: add transmission elements
3. panda.gazebo.xacro: add gazebo_ros_control plugin

## use panda_erg
To use the KDL parser in the panda_erg package we had to convert the panda_arm_hand.urdf.xacro to the panda_arm_hand.urdf