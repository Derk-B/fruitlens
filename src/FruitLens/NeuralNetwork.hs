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
  , main
  ) where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS

import Control.Monad
import Data.Ord
import Data.List ( foldl', maximumBy, transpose )
import System.Random


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

-- | Derivative of the activation function for the backpropagation
activation' :: Float -> Float
activation' x | x > 0  = 1
              | otherwise = 0

-- | Calculate the output of a single layer
calculateLayerOutput :: [Float] -> Layer -> [Float]
calculateLayerOutput inputs (biases, weights) = 
  map activation $ zipWith (+) biases $ sum . zipWith (*) inputs <$> weights

-- | Feed forward through the entire neural network
feedForward :: [Float] -> NeuralNetwork -> [Float]
feedForward = foldl' calculateLayerOutput

-- | Calculate the error at the output layer
outputError :: [Float] -> [Float] -> [Float]
outputError target output = zipWith (*) (zipWith (-) target output) (map activation' output)

-- | Calculate the error at a hidden layer
hiddenError :: [Float] -> Weights -> [Float] -> [Float]
hiddenError nextError weights currentOutput = 
  zipWith (*) (map sum $ transpose $ zipWith (\w e -> map (* e) w) weights nextError) (map activation' currentOutput)

backpropagate :: [Float] -> [Float] -> NeuralNetwork -> ([Float], [Layer])
backpropagate input target network = 
  let -- Forward pass: compute outputs for each layer
      outputs = scanl calculateLayerOutput input network
      output = last outputs
      
      -- Compute output error
      outputErrors = outputError target output
      
      -- Backward pass: compute errors and deltas for each layer
      (_, deltas) = foldr (\(layer, layerOutput) (nextErrors, acc) -> 
        let -- Compute errors for the current layer
            errors = hiddenError nextErrors (snd layer) layerOutput
            -- Compute deltas for the current layer
            deltaBiases = errors
            deltaWeights = map (\e -> map (* e) layerOutput) errors
        in (errors, (deltaBiases, deltaWeights) : acc))
        (outputErrors, []) (zip network (init outputs))
  in (output, deltas)

updateNetwork :: NeuralNetwork -> [Layer] -> Float -> NeuralNetwork
updateNetwork network deltas learningRate = 
  zipWith (\(biases, weights) (deltaBiases, deltaWeights) -> 
    (zipWith (-) biases (map (* learningRate) deltaBiases), 
     zipWith (\w dw -> zipWith (-) w (map (* learningRate) dw)) weights deltaWeights))
    network deltas

prettyPrint :: NeuralNetwork -> String
prettyPrint [] = []
prettyPrint ((bs,ws):ls) = concat (zipWith (\b w -> show b ++ " | " ++ show w ++ "\n") bs ws) ++ "\n" ++ prettyPrint ls

train :: [Float] -> [Float] -> NeuralNetwork -> Float -> IO NeuralNetwork
train input target network learningRate = do
  let (output, deltas) = backpropagate input target network
  return $ updateNetwork network deltas learningRate

getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]
getX     s n = (/ 256) <$> getImage s n
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY     s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]


main :: IO ()
main = do
  [trainI, trainL, testI, testL] <- mapM ((decompress  <$>) . BS.readFile)
    [ "train-images-idx3-ubyte.gz"
    , "train-labels-idx1-ubyte.gz"
    ,  "t10k-images-idx3-ubyte.gz"
    ,  "t10k-labels-idx1-ubyte.gz"
    ]
  -- print $ map length [trainI, trainL, testI, testL]
  network <- newModel [784, 30, 10]
  
  -- Train the network
  trainedNetwork <- foldM (\n i -> train (getX trainI i) (getY trainL i) n 0.02 ) network [0..9999]
  
  let  bestOf = fst . maximumBy (comparing snd) . zip [0..]
  let guesses = bestOf . (\n -> feedForward (getX testI n) trainedNetwork) <$> [0..9999]
  let answers = getLabel testL <$> [0..9999]
  putStrLn $ show (sum $ fromEnum <$> zipWith (==) guesses answers) ++
    " / 10000"

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