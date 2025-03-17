module FruitLens.NeuralNetwork
  ( Biases
  , Weights
  , Layer
  , NeuralNetwork
  , newModel
  , activation
  , calculateLayerOutput
  , feedForward
  , predictFruit
  , trainModel
  , FruitType(..)
  ) where

import Control.Monad
import Data.Functor
import Data.Ord
import Data.List
import System.Random

import FruitLens.Utils (gauss)

-- | Fruit types that can be recognized
data FruitType = Apple | Banana | Orange | Strawberry | Grape
  deriving (Show, Eq, Enum, Bounded)

-- | Convert FruitType to string representation
fruitTypeToString :: FruitType -> String
fruitTypeToString Apple = "apple"
fruitTypeToString Banana = "banana"
fruitTypeToString Orange = "orange"
fruitTypeToString Strawberry = "strawberry"
fruitTypeToString Grape = "grape"

-- | Type aliases for neural network components
type Biases = [Float]
type Weights = [[Float]]
type Layer = (Biases, Weights)
type NeuralNetwork = [Layer]

-- | Create a new neural network model with the given layer sizes
-- The first element is the number of inputs, and the last element
-- is the number of outputs. Elements in between are hidden layer sizes.
newModel :: [Int] -> IO NeuralNetwork
newModel [] = error "newModel: cannot initialize layers with [] as input"
newModel layers@(_:outputLayers) = do
  biases <- mapM (\n -> replicateM n (randomRIO (-1,1) :: IO Float)) outputLayers
  weights <- zipWithM (\m n -> replicateM n $ replicateM m (randomRIO (-1,1) :: IO Float)) layers outputLayers
  return (zip biases weights)

-- | Activation function (ReLU)
activation :: Float -> Float
activation x | x > 0      = x
             | otherwise  = 0

-- | Sigmoid activation function for output layer
sigmoid :: Float -> Float
sigmoid x = 1 / (1 + exp (-x))

-- | Calculate the output of a single layer
calculateLayerOutput :: [Float] -> Layer -> [Float]
calculateLayerOutput inputs (biases, weights) = 
  map sigmoid $ zipWith (+) biases $ sum . zipWith (*) inputs <$> weights

-- | Feed forward through the entire neural network
feedForward :: [Float] -> NeuralNetwork -> [Float]
feedForward = foldl' calculateLayerOutput

loss :: [Float] -> [Float] -> Float
loss values targets = sum $ zipWith (\ a b -> (b - a) * (b - a)) values targets

modelLoss :: NeuralNetwork -> [Float] -> [Float] -> Float
modelLoss m ins = loss (feedForward ins m)

-- Gradient descent sensitivity
gdSens :: Float
gdSens = 0.01

updateAt :: (a -> a) -> Int -> [a] -> [a]
updateAt f p ls = l ++ [f (head r)] ++ tail r
  where (l,r) = splitAt p ls

updateModel :: NeuralNetwork -> NeuralNetwork -> NeuralNetwork
updateModel [] _ = []
updateModel oldModel@(oldLayer@(oldBias,oldWeights):nextLayers) prevLayers = newLayer : updateModel nextLayers (oldLayer: prevLayers)
  where
    targets = [1,2,3]
    testInput = [1,1,1]
    originalCost = loss (feedForward testInput oldModel) targets
    costGradientB :: [Float]
    costGradientB = let
      -- Adds gsSens to each bias, example: [1,2,3] -> [[1+gdSens,2,3],[1,2+gdSens,3],...]
      updatedBiases :: [[Float]]
      updatedBiases = map (\i -> updateAt (+ gdSens) i oldBias) ([0..(length oldBias)] :: [Int]) 
      in map (\newBias -> modelLoss (prevLayers ++ [(newBias, oldWeights)] ++ nextLayers) testInput targets / gdSens) updatedBiases

    costGradientW :: [[Float]]
    costGradientW = let
      -- Weights is an MxN array, so we need a MxN number of updated weight arrays
      updateIndices = [(x,y) | y <- [0..(length oldWeights)], x <- [0..(length (head oldWeights))]]
      -- M x N weights 
      updatedWeights ::[[[Float]]]
      updatedWeights = map (\(y,x) -> updateAt (updateAt (+ gdSens) x) y oldWeights) updateIndices
      in map (\newWeights -> map (\newWeight -> modelLoss (prevLayers ++ [(oldBias, newWeights)] ++ nextLayers) testInput targets / gdSens) newWeights) updatedWeights
    learningRate = 0.001

    -- Gradient descent for a single layer
    newLayer = (
        -- Apply the bias gradients to the old biases
        zipWith (\b g -> b - (g * learningRate) ) oldBias  costGradientB, 
        -- Apply the weight gradients to the old weights
        -- Weigth and weightGradients are a 2d array so apply each weight in each weightArray to a gradient in each gradientArray
        -- [[1,2,3],...] <- old weights
        --   | | |
        -- [[1,1,1],...] <- gradients
        --   | | |
        -- [[a,b,c],...] <- new weights
        zipWith (zipWith (\ w g -> w - (g * learningRate))) oldWeights costGradientW
      )

prettyPrint :: NeuralNetwork -> String
prettyPrint [] = []
prettyPrint ((bs,ws):ls) = concat (zipWith (\b w -> show b ++ " | " ++ show w ++ "\n") bs ws) ++ "\n" ++ prettyPrint ls

-- bTest = res where
--   oldBs = [1,1,1,1]
--   updatedBiases :: [[Float]]
--   updatedBiases = map (\i -> updateAt (+ gdSens) i oldBs) ([0..length oldBs-1] :: [Int]) 
--   res = updatedBiases

-- wTest = res where
--   oldWs = [[1,1,1],[1,1,1]]
--   -- Weights is an MxN array, so we need a MxN number of updated weight arrays
--   updateIndices = [(x,y) | y <- [0..length oldWs-1], x <- [0..length (head oldWs)-1]]
--   -- M x N weights 
--   updatedWeights ::[[[Float]]]
--   updatedWeights = map (\(y,x) -> updateAt (updateAt (+ gdSens) x) y oldWs) updateIndices
--   res = updatedWeights

start = do
  -- model <- newModel [784, 30, 10]
  let targets = [1,2,3]
  model <- newModel [3,4,3]
  putStrLn "========================="
  putStrLn "Model:"
  putStrLn $ prettyPrint model
  putStrLn "========================="

  let result = feedForward [1,1,1] model
  putStrLn "Result:"
  print result
  putStrLn "Loss:"
  print $ loss result targets

  let updatedModel = updateModel model []
  let updatedResult = feedForward [1,1,1] updatedModel
  putStrLn "========================="
  putStrLn "Model:"
  putStrLn $ prettyPrint updatedModel
  putStrLn "========================="
  putStrLn "Result:"
  print updatedResult
  putStrLn "Loss:"
  print $ loss updatedResult targets
  
  let updatedModel2 = updateModel updatedModel []
  let updatedResult2 = feedForward [1,1,1] updatedModel2
  putStrLn "========================="
  putStrLn "Model:"
  putStrLn $ prettyPrint updatedModel2
  putStrLn "========================="
  putStrLn "Result:"
  print updatedResult2
  putStrLn "Loss:"
  print $ loss updatedResult2 targets

-- | Extract features from image data
-- Each pixel is represented as [r,g,b] and the image is a 2D array of pixels
extractFeatures :: [[[Float]]] -> [Float]
extractFeatures imageData = 
  -- Simple feature extraction: average RGB values
  let totalPixels = length imageData * (if null imageData then 0 else length (head imageData))
      sumRGB = foldl' (\(r,g,b) row -> 
                foldl' (\(r',g',b') pixel -> 
                        case pixel of
                          [r'',g'',b''] -> (r'+r'', g'+g'', b'+b'')
                          _ -> (r', g', b')) 
                       (r,g,b) row) 
                (0,0,0) imageData
      (avgR, avgG, avgB) = if totalPixels == 0 
                           then (0,0,0) 
                           else let fp = fromIntegral totalPixels
                                in (fst3 sumRGB / fp, snd3 sumRGB / fp, thd3 sumRGB / fp)
      -- Calculate color variance as additional features
      varRGB = foldl' (\(vr,vg,vb) row ->
                foldl' (\(vr',vg',vb') pixel -> 
                        case pixel of
                          [r,g,b] -> (vr' + (r - avgR)^2, 
                                      vg' + (g - avgG)^2, 
                                      vb' + (b - avgB)^2)
                          _ -> (vr', vg', vb')) 
                       (vr,vg,vb) row)
                (0,0,0) imageData
      (varR, varG, varB) = if totalPixels == 0 
                           then (0,0,0) 
                           else let fp = fromIntegral totalPixels
                                in (fst3 varRGB / fp, snd3 varRGB / fp, thd3 varRGB / fp)
  in [avgR, avgG, avgB, varR, varG, varB]
  where
    fst3 (a,_,_) = a
    snd3 (_,b,_) = b
    thd3 (_,_,c) = c

-- | Predict the fruit type from image data
predictFruit :: [[[Float]]] -> String
predictFruit imageData = 
  -- In a real implementation, we would:
  -- 1. Load a pre-trained model
  -- 2. Extract features from the image
  -- 3. Run the features through the model
  -- 4. Return the predicted fruit type
  
  -- For now, we'll use a simple heuristic based on color
  let features = extractFeatures imageData
      [avgR, avgG, avgB, _, _, _] = features
  in if null imageData 
     then "unknown"
     else if avgR > avgG && avgR > avgB 
          then fruitTypeToString Apple
          else if avgG > avgR && avgG > avgB 
               then fruitTypeToString Banana
               else if avgR > avgB && avgG > avgB && abs (avgR - avgG) < 0.2
                    then fruitTypeToString Orange
                    else if avgR > avgG && avgR > avgB && avgB > avgG
                         then fruitTypeToString Strawberry
                         else if avgB > avgR && avgB > avgG
                              then fruitTypeToString Grape
                              else "unknown"


-- | Train the neural network model (placeholder)
-- In a real implementation, this would use backpropagation
trainModel :: NeuralNetwork -> [([Float], [Float])] -> Int -> Float -> IO NeuralNetwork
trainModel model _ 0 _ = return model
trainModel model trainingData epochs learningRate = do
  -- Placeholder for actual training logic
  putStrLn $ "Training model: " ++ show epochs ++ " epochs remaining"
  -- In a real implementation, we would:
  -- 1. For each training example:
  --    a. Perform forward pass
  --    b. Calculate error
  --    c. Backpropagate error
  --    d. Update weights and biases
  -- 2. Repeat for specified number of epochs
  trainModel model trainingData (epochs - 1) learningRate