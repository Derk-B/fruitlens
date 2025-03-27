\begin{code}

{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}

module AI where

import Codec.Compression.GZip (decompress)
import Control.Monad
import qualified Data.ByteString.Lazy as BS
import Data.List (foldl', maximumBy, transpose)
import Data.Ord
import qualified GHC.Int
import System.Random
import Utils (gauss)
import Data.Binary (encode, decode)
import qualified Data.ByteString.Lazy as BL
import System.Directory (doesFileExist)

-- Fruit types that can be recognized by the neural network
data FruitType = Apple | Banana | Pear
  deriving (Show, Eq, Enum, Bounded)

type Biases = [Float]
type PoolSize = Int
type Weights = [[Float]]
type FullyConnectedLayer = (Biases, Weights)
type Kernel = [[Float]]
type Image = [[[Float]]]

data Layer = ConvLayer ConvolutionalLayer
           | MaxPoolingLayer PoolSize
           | FullyConnected FullyConnectedLayer
           deriving (Eq)
type ConvolutionalLayer = ([Kernel], Biases)
type NeuralNetwork = [Layer]

reLuactivation :: Float -> Float
reLuactivation x | x > 0     = x
                 | otherwise = 0

reLuDerivative :: Float -> Float
reLuDerivative x | x > 0     = 1
                 | otherwise = 0

softmax :: [Float] -> [Float]
softmax xs =
  let expXs = map exp xs
      sumExpXs = sum expXs
  in map (/ sumExpXs) expXs

crossEntropyLoss :: [Float] -> [Float] -> Float
crossEntropyLoss predicted target = sum $ zipWith (\t p -> if t > 0 then -(t * log p) else 0) target (map (max 1e-15 . min (1 - 1e-15)) predicted)

crossEntropyDerivative :: [Float] -> [Float] -> [Float]
crossEntropyDerivative = zipWith (-)

convolve :: Image -> Kernel -> [[Float]]
convolve img kernel =
  let kRows      = length kernel
      kCols      = length (head kernel)
      iRows      = length img
      iCols      = length (head img)
      numChannels = length (head (head img))
  in [[sum [sum [(kernel !! ki !! kj) * (img !! (i + ki) !! (j + kj) !! c)
                   | c <- [0 .. numChannels - 1]]
           | ki <- [0 .. kRows - 1]
           , kj <- [0 .. kCols - 1]]
       | j <- [0 .. iCols - kCols]]
     | i <- [0 .. iRows - kRows]]

combineFeatureMaps :: [[[Float]]] -> Image
combineFeatureMaps featureMaps =
  let h = length (head featureMaps)
      w = length (head (head featureMaps))
  in [[[fm !! i !! j | fm <- featureMaps]
       | j <- [0 .. w - 1]]
       | i <- [0 .. h - 1]]

applyConvLayer :: Image -> ConvolutionalLayer -> Image
applyConvLayer img (kernels, biases) =
  let featureMaps = zipWith (\kernel bias ->
                        let convMap = convolve img kernel
                        in map (map (\x -> reLuactivation (x + bias))) convMap
                      ) kernels biases
  in combineFeatureMaps featureMaps

applyMaxPoolingLayer :: Image -> PoolSize -> Image
applyMaxPoolingLayer img poolSize =
  let height   = length img
      width    = length (head img)
      channels = length (head (head img))
      pooledH  = height `div` poolSize
      pooledW  = width `div` poolSize
      maxPool i j = [maximum[img !! (i + di) !! (j + dj) !! c
                              | di <- [0 .. poolSize - 1]
                              , dj <- [0 .. poolSize - 1]]
                    | c <- [0 .. channels - 1]]
  in [[maxPool (i * poolSize) (j * poolSize)
       | j <- [0 .. pooledW - 1]]
       | i <- [0 .. pooledH - 1]]

flattenImage :: Image -> [Float]
flattenImage = concatMap concat

calculateFullyConnectedLayerOutput :: [Float] -> FullyConnectedLayer -> [Float]
calculateFullyConnectedLayerOutput inputs (biases, weights) =
  map reLuactivation $ zipWith (+) biases $ map (sum . zipWith (*) inputs) weights

feedForwardImage :: Image -> NeuralNetwork -> [Float]
feedForwardImage img (layer:layers) =
  case layer of
    ConvLayer conv       -> feedForwardImage (applyConvLayer img conv) layers
    MaxPoolingLayer size -> feedForwardImage (applyMaxPoolingLayer img size) layers
    -- Fully connected layer does not reconstruct an Image type for recursion
    -- but returns the final [Float] after computing all the fc layers using a foldl.
    FullyConnected _     -> feedForwardFullyConnected (flattenImage img) (layer:layers)

feedForwardFullyConnected :: [Float] -> NeuralNetwork -> [Float]
feedForwardFullyConnected =
  foldl (\acc layer ->
           case layer of
             FullyConnected fc -> calculateFullyConnectedLayerOutput acc fc
             _ -> error "feedForwardFullyConnected: Expected only fully connected layers."
        )

randomKernel :: Int -> Int -> IO Kernel
randomKernel i j = replicateM i (replicateM j (gauss 0.001))

newModel :: IO NeuralNetwork
newModel = do
  -- First convolutional layer: 8 kernels (3×3)
  conv1Kernels <- replicateM 8 (randomKernel 3 3)
  conv1Biases  <- replicateM 8 (gauss 0.01)
  let convLayer1 = ConvLayer (conv1Kernels, conv1Biases)

  -- First max pooling layer with pool size 2×2
  let poolLayer1 = MaxPoolingLayer 2

  -- Second convolutional layer: 16 3x3 kernels
  conv2Kernels <- replicateM 16 (randomKernel 3 3)
  conv2Biases  <- replicateM 16 (gauss 0.01)
  let convLayer2 = ConvLayer (conv2Kernels, conv2Biases)

  -- Second max pooling layer with pool size 2×2
  let poolLayer2 = MaxPoolingLayer 2

  -- Fully connected layer 1: 8464 -> 100
  fc1Biases  <- replicateM 100 (gauss 0.01)
  fc1Weights <- replicateM 100 (replicateM 8464 (gauss 0.01))
  let fcLayer1 = FullyConnected (fc1Biases, fc1Weights)

  -- Fully connected layer 2: 100 -> 5 (one for each fruit type)
  fc2Biases  <- replicateM 5 (gauss 0.01)
  fc2Weights <- replicateM 5 (replicateM 100 (gauss 0.01))
  let fcLayer2 = FullyConnected (fc2Biases, fc2Weights)

  return [convLayer1, poolLayer1, convLayer2, poolLayer2, fcLayer1, fcLayer2]

forwardPass :: Image -> NeuralNetwork -> ([Float], [Image])
forwardPass inputImage network =
  let (outputs, images) = foldl propagateLayer ([], [inputImage]) network
      finalOutput = head outputs
  in (finalOutput, images)
  where
    propagateLayer (outputs, images@(prevImage:_)) layer =
      case layer of
        ConvLayer convLayer ->
          let newImage = applyConvLayer prevImage convLayer
          in (outputs, newImage : images)
        MaxPoolingLayer poolSize ->
          let newImage = applyMaxPoolingLayer prevImage poolSize
          in (outputs, newImage : images)
        FullyConnected fcLayer ->
          let flatInput = flattenImage prevImage
              layerOutput = calculateFullyConnectedLayerOutput flatInput fcLayer
          in (softmax layerOutput : outputs, images)

backpropFullyConnected :: Float -> [Float] -> [Float] -> [Float] -> ((Biases, Weights), [Float])
backpropFullyConnected learningRate inputs target layerOutput =
  let errorDerivative = crossEntropyDerivative layerOutput target

      activationDerivatives = map reLuDerivative layerOutput

      delta = zipWith (*) errorDerivative activationDerivatives

      biasUpdates = map (learningRate *) delta

      weightUpdates = [map ((learningRate *) . (x *)) delta | x <- inputs]

      propagatedError = [sum $ zipWith (*) delta row | row <- transpose weightUpdates]
  in ((biasUpdates, weightUpdates), propagatedError)

trainIteration :: NeuralNetwork -> (Image, [Float]) -> Float -> IO NeuralNetwork
trainIteration model (inputImage, targetOutput) learningRate = do
  let (outputs, intermediateImages) = forwardPass inputImage model
  let initialError = crossEntropyDerivative outputs targetOutput

  (updatedModel, _) <- foldM (backpropagateLayer learningRate) (model, initialError) (zip (reverse model) (reverse intermediateImages))
  return updatedModel
  where
    backpropagateLayer :: Float -> (NeuralNetwork, [Float]) -> (Layer, Image) -> IO (NeuralNetwork, [Float])
    backpropagateLayer lr (currentModel, errorToPropagate) (layer, layerInput) =
      case layer of
        FullyConnected fcLayer -> do
          let ((biasUpdates, weightUpdates), newErrorToPropagate) = backpropFullyConnected lr (flattenImage layerInput) targetOutput (calculateFullyConnectedLayerOutput (flattenImage layerInput) fcLayer)
          let updatedLayer = FullyConnected (biasUpdates, weightUpdates)
          return (replaceLayer currentModel layer updatedLayer, newErrorToPropagate)

        _ -> return (currentModel, errorToPropagate)

replaceLayer :: NeuralNetwork -> Layer -> Layer -> NeuralNetwork
replaceLayer [] _ _ = []
replaceLayer (l:ls) oldLayer newLayer
  | l == oldLayer = newLayer : ls
  | otherwise     = l : replaceLayer ls oldLayer newLayer

trainModel :: NeuralNetwork -> [(Image, [Float])] -> Int -> Float -> IO NeuralNetwork
trainModel initialModel trainingData epochs learningRate = do
  foldM trainEpoch initialModel [1..epochs]
  where
    trainEpoch model epoch = do
      putStrLn $ "Epoch " ++ show epoch ++ "/" ++ show epochs
      foldM (\currentModel img -> trainIteration currentModel img learningRate) model trainingData

evaluateModel :: NeuralNetwork -> [(Image, [Float])] -> IO ()
evaluateModel trainedModel testData = do
    let testResults = map (\(img, label) ->
            let prediction = predictFruit trainedModel img
                expectedFruit = case label of
                    [1.0, 0.0, 0.0] -> Apple
                    [0.0, 1.0, 0.0] -> Banana
                    [0.0, 0.0, 1.0] -> Pear
                    _ -> error "Invalid label"
            in prediction == expectedFruit
            ) testData

    let accuracy = ((fromIntegral (length (filter id testResults)) / fromIntegral (length testResults)) * 100) :: Double

    putStrLn $ "Test Accuracy: " ++ show accuracy ++ "%"

argmax :: [Float] -> Int
argmax xs = snd $ maximumBy (comparing fst) (zip xs [0..])

predictFruit :: NeuralNetwork -> Image -> FruitType
predictFruit model image = toEnum (argmax (feedForwardImage image model))

-- Model Persistence
-- saveModel :: FilePath -> NeuralNetwork -> IO ()
-- saveModel filePath model = BL.writeFile filePath (encode model)

-- loadModel :: FilePath -> IO NeuralNetwork
-- loadModel filePath = decode <$> BL.readFile filePath

\end{code}