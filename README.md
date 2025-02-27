# FruitLens
FruitLens is an app that can detect different types of fruit from the camera feed of mobile phones.

## Project structure
FruitLens requires an app that works on Android and IOS that sends the camera feed to a server that is running the object recognition software (ORS). 

The server should run an api that will receive the camera feed and sends it to the ORS.

The ORS will be written in haskell and will be the main focus of this project.

## Requirements
This project uses *stack* to build and run. The easiest way to install this is using *ghcup*.

This project runs on *GHC 9.4.8*. If you don't have this installed, then stack should automatically install this GHC version when you run the project.

## How to run
Download all the files from https://github.com/golbin/TensorFlow-MNIST/tree/master/mnist/data

Put these in the root of the project `numberguesser/<files here>`

Then run `cd` into `numberguesser/` and type `stack run` and you should see the training progress. It will run for about 1 minute and then show the accuracy of the model.