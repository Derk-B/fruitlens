\hide {
\begin{code}
{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}

module AI where

import Codec.Compression.GZip (decompress)
import Control.Monad
import qualified Data.ByteString.Lazy as BS
import Data.List (maximumBy, transpose)
import Data.Ord
import qualified GHC.Int
import System.Random
import Utils (gauss, gaussian)
import Data.Binary (encode, decode)
import qualified Data.ByteString.Lazy as BL
import System.Directory (doesFileExist)
\end{code}
}


\begin{figure}
  \centering
  \includegraphics[width=0.99\linewidth]{assets/conv.png}
  \caption{Example of a convolution being applied to an image.}
\end{figure}

Becasue the goal of this project is to recognise fruits in images, we can exploit
the spatial structure of these images. As pixels are related to their neighbours,
we can use convolutional filters to extract spatial information of a region in the
input image. The resulting value will therefore encode information about the value of
a given pixel, as well as its neighbouring pixels. This will make the model more robust against
images that are shifted or rotated as compared to the images in the training set.
\begin{code}
-- FruitType lists the types of fruit that can be recognised by the model.
data FruitType = Apple | Banana
  deriving (Show, Eq, Enum, Bounded)
\end{code}

In order to make the code of the neural network more readable, types are
\begin{code}
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

-- randomKernel :: Int -> Int -> IO Kernel
-- randomKernel i j = replicateM i (replicateM j (gauss 0.001))
-- https://staff.fnwi.uva.nl/r.vandenboomgaard/ComputerVision/LectureNotes/IP/LocalStructure/GaussianDerivatives.html
randomKernel :: Int -> Float -> Kernel
randomKernel size sigma = map (map (/ total)) kernel
  where
    center = fromIntegral (size `div` 2)
    kernel = [[gaussian (fromIntegral i - center) (fromIntegral j - center) sigma
                 | j <- [0 .. size - 1]]
               | i <- [0 .. size - 1]]
    total = sum (map sum kernel)

\end{code}

\begin{figure}
  \centering
  \includegraphics[width=0.99\linewidth]{assets/cnn.jpg}
  \caption{Example of the architecture of a Convolutional Neural Network.}
\end{figure}

\begin{code}
-- WIP with CNN
newModelCNN :: IO NeuralNetwork
newModelCNN = do
  -- First convolutional layer: 8 kernels (3×3)
  let conv1Kernels = replicateM 8 (randomKernel 3 1.0)
  conv1Biases  <- replicateM 8 (gauss 0.01)
  let convLayer1 = ConvLayer (conv1Kernels, conv1Biases)

  -- First max pooling layer with pool size 2×2
  let poolLayer1 = MaxPoolingLayer 2

  -- Second convolutional layer: 16 3x3 kernels
  let conv2Kernels = replicateM 16 (randomKernel 3 1.0)
  conv2Biases  <- replicateM 16 (gauss 0.01)
  let convLayer2 = ConvLayer (conv2Kernels, conv2Biases)

  -- Second max pooling layer with pool size 2×2
  let poolLayer2 = MaxPoolingLayer 2

  -- Fully connected layer 1: 8464 -> 100
  fc1Biases  <- replicateM 100 (gauss 0.01)
  fc1Weights <- replicateM 100 (replicateM 8464 (gauss 0.01))
  let fcLayer1 = FullyConnected (fc1Biases, fc1Weights)

  -- Fully connected layer 2: 100 -> 2 (one for each fruit type)
  fc2Biases  <- replicateM 2 (gauss 0.01)
  fc2Weights <- replicateM 2 (replicateM 100 (gauss 0.01))
  let fcLayer2 = FullyConnected (fc2Biases, fc2Weights)

  return [convLayer1, poolLayer1, convLayer2, poolLayer2, fcLayer1, fcLayer2]

-- Model without convolutional layers
newModelFC :: IO NeuralNetwork
newModelFC = do
  -- Fully connected layer 1: 30000 -> 3000
  fc1Biases  <- replicateM 100 (gauss 0.01)
  fc1Weights <- replicateM 100 (replicateM 30000 (gauss 0.01))
  let fcLayer1 = FullyConnected (fc1Biases, fc1Weights)

  -- Fully connected layer 1: 3000 -> 1000
  fc2Biases  <- replicateM 2 (gauss 0.01)
  fc2Weights <- replicateM 2 (replicateM 100 (gauss 0.01))
  let fcLayer2 = FullyConnected (fc2Biases, fc2Weights)

  return [fcLayer1, fcLayer2]

forwardPass :: Image -> NeuralNetwork -> ([Float], [Image])
forwardPass inputImage network =
  let (outputs, images) = foldl propagateLayer ([], [inputImage]) network
      finalOutput = head outputs
  in (finalOutput, images)
  where
    propagateLayer (outputs, images@(prevImage:_)) layer =
      case layer of
        ConvLayer convLayer -> (outputs, applyConvLayer prevImage convLayer : images)
        MaxPoolingLayer poolSize -> (outputs, applyMaxPoolingLayer prevImage poolSize : images)
        FullyConnected fcLayer -> (softmax (calculateFullyConnectedLayerOutput (flattenImage prevImage) fcLayer) : outputs, images)

backpropFullyConnected :: Float -> [Float] -> [Float] -> (Biases, Weights) -> ((Biases, Weights), [Float])
backpropFullyConnected learningrate inputs propagatedError (biases, weights) =
  let layerOutput = calculateFullyConnectedLayerOutput inputs (biases, weights)
      -- Compute the derivative of the activation function (ReLU).
      activationDerivatives = map reLuDerivative layerOutput
      -- Compute the delta for this layer by multiplying the propagated error
      -- with the derivative.
      delta = zipWith (*) propagatedError activationDerivatives
      -- Compute gradients for biases and weights.
      biasGradients = map (learningrate *) delta
      weightGradients = [[learningrate * x * d | d <- delta] | x <- inputs]
      -- Update parameters by subtracting the gradients.
      newBiases = zipWith (-) biases biasGradients
      newWeights = zipWith (zipWith (-)) weights weightGradients
      -- Propagate error to the previous layer using the original weights.
      newPropagatedError = [sum $ zipWith (*) delta col | col <- transpose weights]
  in ((newBiases, newWeights), newPropagatedError)

trainIteration :: NeuralNetwork -> (Image, [Float]) -> Float -> IO NeuralNetwork
trainIteration model (inputImage, targetOutput) learningRate = do
  let (outputs, intermediateImages) = forwardPass inputImage model
      initialError = crossEntropyDerivative outputs targetOutput
  (updatedModel, _) <- foldM (backpropagateLayer learningRate) (model, initialError) (zip (reverse model) (reverse intermediateImages))
  return updatedModel
  where
    backpropagateLayer :: Float -> (NeuralNetwork, [Float]) -> (Layer, Image) -> IO (NeuralNetwork, [Float])
    backpropagateLayer lr (currentModel, errorToPropagate) (layer, layerInput) =
      case layer of
        FullyConnected fcLayer -> do
          let flatInput = flattenImage layerInput
              (updatedFC, newError) = backpropFullyConnected lr flatInput errorToPropagate fcLayer
          return (replaceLayer currentModel layer (FullyConnected updatedFC), newError)
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
                    [1.0, 0.0] -> Apple
                    [0.0, 1.0] -> Banana
                 --   [0.0, 0.0, 1.0] -> Pear
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