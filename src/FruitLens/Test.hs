module FruitLens.Test where

import Control.Monad
import Data.Functor
import Data.Ord
import Data.List
import System.Random

import FruitLens.Utils (gauss)

-- Type aliases for neural network components
type Biases = [Float]
type Weights = [[Float]]
type Layer = (Biases, Weights)
type NeuralNetwork = [Layer]

-- | Create a new neural network model with the given layer sizes
newModel :: [Int] -> IO NeuralNetwork
newModel [] = error "newModel: cannot initialize layers with [] as input"
newModel layers@(_:outputLayers) = do
  biases <- mapM (\n -> replicateM n (gauss 0.01)) outputLayers
  weights <- zipWithM (\m n -> replicateM n $ replicateM m $ gauss 0.01) layers outputLayers
  return (zip biases weights)

-- | Activation function (ReLU)
activation :: Float -> Float
activation x | x > 0      = x
             | otherwise  = 0

-- | Derivative of ReLU
activation' :: Float -> Float
activation' x | x > 0      = 1
              | otherwise   = 0

-- | Sigmoid activation function for output layer
sigmoid :: Float -> Float
sigmoid x = 1 / (1 + exp (-x))

-- | Derivative of sigmoid
sigmoid' :: Float -> Float
sigmoid' x = sigmoid x * (1 - sigmoid x)

-- | Calculate the output of a single layer
calculateLayerOutput :: [Float] -> Layer -> [Float]
calculateLayerOutput inputs (biases, weights) = 
  map sigmoid $ zipWith (+) biases $ sum . zipWith (*) inputs <$> weights

-- | Feed forward through the entire neural network
feedForward :: [Float] -> NeuralNetwork -> [Float]
feedForward = foldl' calculateLayerOutput

-- | Cross-entropy loss function
crossEntropyLoss :: [Float] -> [Float] -> Float
crossEntropyLoss outputs targets = 
    let costPerNode v t = (v - t) * (v - t)
    in - sum (zipWith costPerNode outputs targets) 
--   let -- Compute the log of each output (with a small epsilon to avoid log(0))
--       logOutputs = map (\x -> log (max x 1e-15)) outputs
--       -- Compute the log of (1 - output) for each output
--       logOneMinusOutputs = map (\x -> log (max (1 - x) 1e-15)) outputs
--       -- Compute the cross-entropy loss term for each output-target pair
--       lossTerms = zipWith3 (\t o lo -> t * lo + (1 - t) * logOneMinusOutputs) targets outputs logOutputs
--   in -- Sum all the loss terms and negate the result
--      -sum lossTerms

-- | Backpropagation algorithm
backprop :: [Float] -> [Float] -> NeuralNetwork -> (NeuralNetwork, Float)
backprop inputs targets network = 
  let outputs = feedForward inputs network
      loss = crossEntropyLoss outputs targets
      (deltas, gradients) = computeGradients inputs targets network
      updatedNetwork = updateNetwork network gradients
  in (updatedNetwork, loss)

-- | Compute gradients for each layer
computeGradients :: [Float] -> [Float] -> NeuralNetwork -> ([[Float]], [Layer])
computeGradients inputs targets network = 
  let outputs = feedForward inputs network
      outputDeltas = zipWith (\o t -> (o - t) * sigmoid' o) outputs targets
      (deltas, gradients) = computeGradients' inputs network outputDeltas
  in (deltas, gradients)

computeGradients' :: [Float] -> NeuralNetwork -> [Float] -> ([[Float]], [Layer])
computeGradients' inputs [] _ = ([], [])
computeGradients' inputs ((biases, weights):layers) deltas = 
  let layerInputs = if null layers then inputs else feedForward inputs (init layers)
      weightGradients = map (\delta -> map (* delta) layerInputs) deltas
      biasGradients = deltas
      nextDeltas = map sum $ zipWith (*) (transpose weights) (replicate (length (transpose weights)) deltas)
      (restDeltas, restGradients) = computeGradients' inputs layers nextDeltas
  in (deltas : restDeltas, (biasGradients, weightGradients) : restGradients)

-- | Update the network using gradients
updateNetwork :: NeuralNetwork -> [Layer] -> NeuralNetwork
updateNetwork network gradients = 
  zipWith (\(biases, weights) (biasGrads, weightGrads) -> 
    (zipWith (-) biases (map (* learningRate) biasGrads),
     zipWith (zipWith (-)) weights (map (map (* learningRate)) weightGrads)))
  network gradients
  where learningRate = 0.01

-- | Training loop
train :: NeuralNetwork -> [([Float], [Float])] -> Int -> IO NeuralNetwork
train network dataset epochs = foldM (\net _ -> trainEpoch net dataset) network [1..epochs]

trainEpoch :: NeuralNetwork -> [([Float], [Float])] -> IO NeuralNetwork
trainEpoch network dataset = foldM (\net (inputs, targets) -> do
  let (updatedNet, loss) = backprop inputs targets net
  return updatedNet) network dataset

-- Example usage
start = do
  model <- newModel [784, 30, 10]
  -- Assuming you have a dataset :: [([Float], [Float])]
  let dataset = [([1,2,2], [2,2,1])]
  trainedModel <- train model dataset 10
  -- Now you can use trainedModel for predictions
  print "Start"