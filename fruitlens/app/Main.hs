module Main (main) where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS

import Control.Monad
import Data.Functor
import Data.Ord
import Data.List
import System.Random

gauss :: Float -> IO Float
gauss scale = do
  x1 <- randomIO
  x2 <- randomIO
  return $ scale * sqrt (- (2 * log x1)) * cos (2 * pi * x2)

type Biases = [Float]
type Weights = [[Float]]
type Layer = (Biases, Weights)
type NeuralNetwork = [Layer]

-----------------
-- Neural network
-----------------
{-
  Create a new brain with the given number of layers and neurons per layer.
  The first element of the list is the number of inputs, and the last element
  is the number of outputs. The elements in between are the number of neurons in the hidden layers.

  The biases are initialized with 1.
  The weights are initialized with a Gaussian distribution with a standard
  deviation of 0.01.

  Since the weights are initialized using random values, the weights are part of the IO monad.
  That is why we use monadic zip and replicate functions here.
-}
newModel :: [Int] -> IO NeuralNetwork
newModel [] = error "newModel: cannot initialize layers with [] as input"
newModel layers@(_:outputLayers) = do
  let biases = flip replicate 1 <$> outputLayers
  let weights = zipWithM (\m n -> replicateM n $ replicateM m $ gauss 0.01) layers outputLayers
  zip biases <$> weights

-- Activation function
activation :: Float -> Float
activation x | x > 1      = 1
             | otherwise  = 0

-- Calculate the weighted sum of the inputs and the weights, and add the biases.
calculateLayerOutput :: [Float] -> Layer -> [Float]
calculateLayerOutput inputs (biases, weigths) = map activation $ zipWith (+) biases $ sum . zipWith (*) inputs <$> weigths

-- Calculate output of each layer in the neural network
feedForward :: [Float] -> NeuralNetwork -> [Float]
feedForward = foldl' calculateLayerOutput

main :: IO ()
main = do
  initModel <- newModel [784, 30, 10]
  let inputs = [1..784]
  let outputs = feedForward inputs initModel
  print outputs