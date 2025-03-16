\begin{code}
module FruitLens.NeuralNetwork
  ( NeuralNetwork
  , newModel
  , feedForwardImage
  , predictFruit
  , trainModel
  , FruitType(..)
  ) where

import Control.Monad ( replicateM )
import Data.List ( maximumBy )
import Data.Ord ( comparing )
import FruitLens.Utils (gauss)
\end{code}

Fruit types that can be recognized by the neural network.
\begin{code}
data FruitType = Apple | Banana | Orange | Strawberry | Grape
  deriving (Show, Eq, Enum, Bounded)
\end{code}

The fruitTypeToString function converts the types of fruit in the Enum to
readable strings. This is useful when the result needs to be shown in the
terminal or on the app.
\begin{code}
fruitTypeToString :: FruitType -> String
fruitTypeToString Apple      = "apple"
fruitTypeToString Banana     = "banana"
fruitTypeToString Orange     = "orange"
fruitTypeToString Strawberry = "strawberry"
fruitTypeToString Grape      = "grape"

----------------------------------------------------------------------
-- Type defenitions.
----------------------------------------------------------------------
type Biases = [Float]
type PoolSize = Int
type Weights = [[Float]]
type FullyConnectedLayer = (Biases, Weights)
\end{code}

A convolutional layer consists of a list of 2D convolutional kernels and one
bias per kernel that gets added to the result. A MaxPooling layer takes the maximum
value of a nxn pool where n is the poolSize. This shrinks the input down before
going to the next layer. The FullyConnected layer consists of biases and weights.

\begin{figure}
    \centering
    \includegraphics[width=0.99\linewidth]{res/cnn.jpg}
    \caption{Example of the architecture of a Convolutional Neural Network.}
\end{figure}
\begin{code}
type ConvolutionalLayer = ([Kernel], Biases)

data Layer = ConvLayer ConvolutionalLayer
           | MaxPoolingLayer PoolSize
           | FullyConnected FullyConnectedLayer
type NeuralNetwork = [Layer]
\end{code}

An image is a 3D list of pixel values. The three dimensions consist of the rows,
columns and the three RGB channels. A kernel is a 2D convolutional kernel that
is used to convolve the image.
\begin{code}
type Image  = [[[Float]]]
type Kernel = [[Float]]

----------------------------------------------------------------------
-- Convolution and Pooling Operations.
----------------------------------------------------------------------
reLuactivation :: Float -> Float
reLuactivation x | x > 0     = x
                 | otherwise = 0

\end{code}
The convolve function convolves an image to a 2D list of floats that give a 2D
feature map after the convolution. This transforms the 3D input image to a 2D
representation as the channels of the images get converted to a 1D vector.
\begin{code}
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
\end{code}

The combineFeatureMaps function reconstructs an Image type with with the channels
resulting from the convolution.
\begin{code}
combineFeatureMaps :: [[[Float]]] -> Image
combineFeatureMaps featureMaps =
  let h = length (head featureMaps)
      w = length (head (head featureMaps))
  in [[[fm !! i !! j | fm <- featureMaps]
       | j <- [0 .. w - 1]]
       | i <- [0 .. h - 1]]
\end{code}

applyConvLayer applies a convolutional layer to an image. The image gets convolved with the
convolutional kernels of the layer and the bias of the layer gets added to the
result. The result gets reconstructed back to an Image type by the combineFeatureMaps
function.
\begin{code}
applyConvLayer :: Image -> ConvolutionalLayer -> Image
applyConvLayer img (kernels, biases) =
  let featureMaps = zipWith (\kernel bias ->
                        let convMap = convolve img kernel
                        in map (map (\x -> reLuactivation (x + bias))) convMap
                      ) kernels biases
  in combineFeatureMaps featureMaps
\end{code}

Applies a max pooling layer to the image to shrink the image down but keep as
much relevant information by keeping the maximum value of a pooling grid.
\begin{code}
applyMaxPoolingLayer :: Image -> PoolSize -> Image
applyMaxPoolingLayer img poolSize =
  let height   = length img
      width    = length (head img)
      channels = length (head (head img))
      pooledH  = height `div` poolSize
      pooledW  = width `div` poolSize
      -- For each channel, take the maximum over the pool window.
      maxPool i j = [maximum[img !! (i + di) !! (j + dj) !! c
                              | di <- [0 .. poolSize - 1]
                              , dj <- [0 .. poolSize - 1]]
                    | c <- [0 .. channels - 1]]
  in [[maxPool (i * poolSize) (j * poolSize)
       | j <- [0 .. pooledW - 1]]
       | i <- [0 .. pooledH - 1]]
\end{code}

Flattens an Image to a 1D list of floats to be used by the fully connected layer.
\begin{code}
flattenImage :: Image -> [Float]
flattenImage = concatMap concat

----------------------------------------------------------------------
-- Fully Connected Layer.
----------------------------------------------------------------------
\end{code}
Get the output of a fully connected layer by taking a list of input values,
computing the weighted sum for each neuron by multiplying corresponding inputs
and weights and summing them and add the neuron's bias, and then applying the
ReLU activation function to the result. This gives a list of floats, one for each
neuron in the layer.
\begin{code}
calculateFullyConnectedLayerOutput :: [Float] -> FullyConnectedLayer -> [Float]
calculateFullyConnectedLayerOutput inputs (biases, weights) =
  map reLuactivation $ zipWith (+) biases $ map (sum . zipWith (*) inputs) weights
\end{code}

The fully connected feed forward works by processing the list of fully connected
layers. It starts with an empty list and for every fully connected layer it
calculates the sum of the inputs and adds the biases by calling the
calculateFullyConnectedLayerOutput function. The output then gets used as
input in the next iteration. It only works with fully connected layers and
gives an error when it encounters another type of layer.
\begin{code}
feedForwardFullyConnected :: [Float] -> NeuralNetwork -> [Float]
feedForwardFullyConnected =
  foldl (\acc layer ->
           case layer of
             FullyConnected fc -> calculateFullyConnectedLayerOutput acc fc
             _ -> error "feedForwardFullyConnected: Expected only fully connected layers."
        )

----------------------------------------------------------------------
-- Feed forward.
----------------------------------------------------------------------
\end{code}

Forward feed an image through the entire CNN. The convolutional and max pooling
layers are applied on the image until a fully connected layer is encountered,
then the image is flattened and processed as a 1D list of floats to predict
the fruit. The result will be a list of size n_fruits and the index with the
maximum value will be the fruit the model predicted is in the image.
\begin{code}
feedForwardImage :: Image -> NeuralNetwork -> [Float]
feedForwardImage img [] = flattenImage img
feedForwardImage img (layer:layers) =
  case layer of
    ConvLayer conv       -> feedForwardImage (applyConvLayer img conv) layers
    MaxPoolingLayer size -> feedForwardImage (applyMaxPoolingLayer img size) layers
    FullyConnected _     -> feedForwardFullyConnected (flattenImage img) (layer:layers)

----------------------------------------------------------------------
-- Model Initialization.
----------------------------------------------------------------------

-- Randomly initialize a ixj kernel.
randomKernel :: Int -> Int -> IO Kernel
randomKernel i j = replicateM i (replicateM j (gauss 0.001))

-- Architecture:
--   * ConvLayer: 8 kernels of size 3×3
--   * MaxPoolingLayer: pool size 2×2
--   * ConvLayer: 16 kernels of size 3×3
--   * MaxPoolingLayer: pool size 2×2
--   * FullyConnected: from flattened input (23×23×16 = 8464) to 100 neurons
--   * FullyConnected: from 100 neurons to 5 outputs (one per fruit type)
newModel :: IO NeuralNetwork
newModel = do
  -- First convolutional layer: 8 kernels (3×3).
  conv1Kernels <- replicateM 8 (randomKernel 3 3)
  conv1Biases  <- replicateM 8 (gauss 0.01)
  let convLayer1 = ConvLayer (conv1Kernels, conv1Biases)

  -- First max pooling layer with pool size 2×2.
  let poolLayer1 = MaxPoolingLayer 2

  -- Second convolutional layer: 16 3x3 kernels.
  conv2Kernels <- replicateM 16 (randomKernel 3 3)
  conv2Biases  <- replicateM 16 (gauss 0.01)
  let convLayer2 = ConvLayer (conv2Kernels, conv2Biases)

  -- Second max pooling layer with pool size 2×2.
  let poolLayer2 = MaxPoolingLayer 2

  -- After two conv+pool layers:
  --   Input image: 100×100
  --   After convLayer1 (3×3): 98×98 with 8 channels
  --   After poolLayer1 (2×2): floor(98/2) = 49×49 with 8 channels
  --   After convLayer2 (3×3): 47×47 with 16 channels
  --   After poolLayer2 (2×2): floor(47/2) = 23×23 with 16 channels
  --   Flattened size = 23 * 23 * 16 = 8464.

  -- Fully connected layer 1: 8464 -> 100
  fc1Biases  <- replicateM 100 (gauss 0.01)
  fc1Weights <- replicateM 100 (replicateM 8464 (gauss 0.01))
  let fcLayer1 = FullyConnected (fc1Biases, fc1Weights)

  -- Fully connected layer 2: 100 -> 5 (one for each fruit type)
  fc2Biases  <- replicateM 5 (gauss 0.01)
  fc2Weights <- replicateM 5 (replicateM 100 (gauss 0.01))
  let fcLayer2 = FullyConnected (fc2Biases, fc2Weights)

  return [convLayer1, poolLayer1, convLayer2, poolLayer2, fcLayer1, fcLayer2]

-- Get the index of the maximum value of a list.
argmax :: [Float] -> Int
argmax xs = snd $ maximumBy (comparing fst) (zip xs [0..])

-- Predict the fruit type from an image using the CNN.
predictFruit :: NeuralNetwork -> Image -> FruitType
predictFruit model image = toEnum (argmax (feedForwardImage image model))




----------------------------------------------------------------------
-- Training (Placeholder)
----------------------------------------------------------------------
-- In a full implementation, this would include a backpropagation routine.
trainModel :: NeuralNetwork -> [(Image, [Float])] -> Int -> Float -> IO NeuralNetwork
trainModel model _ 0 _ = return model
trainModel model trainingData epochs learningRate = do
  putStrLn $ "Training model: " ++ show epochs ++ " epochs remaining"
  -- Training logic (forward pass, error computation, backpropagation, weight updates) goes here.
  trainModel model trainingData (epochs - 1) learningRate
\end{code}