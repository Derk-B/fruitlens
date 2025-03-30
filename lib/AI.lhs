\hide {
\begin{code}

{-# LANGUAGE DeriveGeneric #-}
{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module AI where

import Control.Monad
import Data.List (maximumBy, transpose)
import Data.Ord
import Data.Binary (encode, decode, Binary)
import GHC.Generics (Generic)
import Utils (gauss, gaussian)
import qualified Data.ByteString.Lazy as BL
\end{code}
}

\begin{figure}
  \centering
  \includegraphics[width=0.5\linewidth]{assets/conv.png}
  \caption{Example of a convolution being applied to an image.}
  \ref{fig:conv}
\end{figure}

Becasue the goal of this project is to recognise fruits in images, we can exploit
the spatial structure of these images. As pixels are related to their neighbours,
we can use convolutional filters to extract spatial information of a region in the
input image. The resulting value will therefore encode information about the value of
a given pixel, as well as its neighbouring pixels. Figure \ref{fig:conv} shows how an input image is convolved using a convolution filter.
The convolution filter produces a new value that takes in information of the neighbourhood in the input image. This will make the model more robust against
images that are shifted or rotated as compared to the images in the training set.
\begin{code}
-- FruitType lists the types of fruit that can be recognised by the model.
data FruitType = Apple | Banana
  deriving (Show, Eq, Enum, Bounded)
\end{code}

In order to make the code of the neural network more readable and generic, types are defined
that define the properties of the neural network. A neural network consists of a
lists of layers, where every layer can be an instance of the following layer types:
\begin{itemize}
  \item \textbf{FullyConnectedLayer:} A fully connected layer consists of biases and weights. These layers are used for the classifying of the fruits in the input images.
  \item \textbf{ConvolutionalLayer:} Performs feature extraction by convolving the input image.
  \item \textbf{MaxPoolingLayer:} Shrinks the output of a convolutional layer down by keeping only the maximum values of a grid with the size of PoolSize.
\end{itemize}
Furthermore, kernels are defined as 2D lists of floats which make up the convolution kernels. Images are defined as a 3D list of floats, as every pixel contains data for the red, green and blue channel.
\begin{code}
type Biases = [Float]
type PoolSize = Int
type Weights = [[Float]]
type Kernel = [[Float]]
type Image = [[[Float]]]
data Layer = ConvLayer ConvolutionalLayer
           | MaxPoolingLayer PoolSize
           | FullyConnected FullyConnectedLayer
           deriving (Eq, Generic)

-- Define the Binary instance for the Layer in for it to be serialized.
instance Binary Layer
type FullyConnectedLayer = (Biases, Weights)
type ConvolutionalLayer = ([Kernel], Biases)
type NeuralNetwork = [Layer]

\end{code}
The activation function adds non-linearity to the model. Without the activation function, the model could only learn linear relationships.
ReLu is used as it is computationaly efficient, it simply outputs the input value if it is greater than 0, and 0 otherwise.
The derivative of the activation function is needed during the backpropogation step of the model.
\begin{code}
reLuactivation :: Float -> Float
reLuactivation x | x > 0     = x
                 | otherwise = 0

reLuDerivative :: Float -> Float
reLuDerivative x | x > 0     = 1
                 | otherwise = 0

\end{code}

The \verb|softMax| function is used in the final layer fully connected layer of the model in order to turn the final layer's output into a probability distribution.
Very large negative values in the output layer get mapped to probability values around 0, values around 0 get mapped to 0.5 and very positive values get mapped to values around 1.
The output of the softMax function will then be used by the argMax function to extract the final prediction. The index of the vector after the softMax function that has the highest value will be picked as the models prediction.
\begin{equation}
s(x_i) = \frac{e^{x_i}}{\sum^{n}_{j=1}{e^{x_j}}}
\end{equation}

\begin{code}
softmax :: [Float] -> [Float]
softmax xs =
  let expXs = map exp xs
      sumExpXs = sum expXs
  in map (/ sumExpXs) expXs

argmax :: [Float] -> Int
argmax xs = snd $ maximumBy (comparing fst) (zip xs [0..])

\end{code}

The \verb|crossEntropyLoss| function computes the cross entropy loss between the predicted probabilities and the true target values, which are encoded as a one-hot vector where only the index corresponding to the correct fruit has a probability of 1.
For example, the one-hot vector for apple is [1, 0], where the first index corresponds to an image containing an apple and therefore has the value 1.
This loss measures the difference between the predicted probability distribution and the actual fruit type.
It penalizes predictions that deviate from the true targets by taking the negative log likelihood of the predicted probability for the correct class.
By clamping the values between $10^{-15}$ and $1 - 10^{-15}$, the case of taking the log of 0 is avoided which would result in an undefined result.

The \verb|crossEntropyDerivative| function calculates the derivative of the loss with respect to the predicted values. This gradient is used during the backpropogation to update the model parameters and minimize the loss.

\begin{code}
crossEntropyLoss :: [Float] -> [Float] -> Float
crossEntropyLoss predicted target = sum $ zipWith (\t p -> if t > 0 then -(t * log p) else 0) target (map (max 1e-15 . min (1 - 1e-15)) predicted)

crossEntropyDerivative :: [Float] -> [Float] -> [Float]
crossEntropyDerivative = zipWith (-)

\end{code}

The \verb|convolve| function convolves an input image with a convolution kernel. This way, information of neighbouring pixels can be taken into account as the result contains information about all the pixels inside of the convolution filter.

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

The \verb|combineFeatureMaps| reconstructs an image type after a convolution. It takes a list of 2D feature maps resulting from the convolution step, each corresponding to a convolution with a different kernel,
and combines them to form an Image type that can then be taken as input for the next layer.

\begin{code}
combineFeatureMaps :: [[[Float]]] -> Image
combineFeatureMaps featureMaps =
  let h = length (head featureMaps)
      w = length (head (head featureMaps))
  in [[[fm !! i !! j | fm <- featureMaps]
       | j <- [0 .. w - 1]]
       | i <- [0 .. h - 1]]

\end{code}

\verb|applyConvLayer| applies a convolutional layer to an image. The image gets convolved with the
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

The \verb|applyMaxPoolingLayer| applies a max pooling layer to the image to shrink the image down but keep as
much relevant information by keeping the maximum value of a pooling grid. The image gets shrunken down based on the poolsize.
For a poolsize of 2, the image gets halved in size. So a 100x100 image will become a 50x50 image after the maxPooling layer.

\begin{code}
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


\end{code}

The \verb|flattenImage| function converts an Image, which is a 3D list of floats, into a 1D list of floats.
This is needed to connect the final maxPooling layer to the input layer of the fully connected network.
\begin{code}
flattenImage :: Image -> [Float]
flattenImage = concatMap concat

\end{code}

\verb|calculateFullyConnectedLayerOutput| gets the output of a fully connected layer by taking a list of input values,
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
layers using a foldl.
For each fully connected layer, it computes the layer's output by using the calculateFullyConnectedLayerOutput function with the current activations.
The resulting output then serves as the input for the next layer.
\begin{code}
feedForwardFullyConnected :: [Float] -> NeuralNetwork -> [Float]
feedForwardFullyConnected =
  foldl (\acc layer ->
           case layer of
             FullyConnected fc -> calculateFullyConnectedLayerOutput acc fc
             _ -> error "feedForwardFullyConnected: Expected only fully connected layers."
        )

\end{code}

The \verb|feedForwardImage| function forward feeds an image through the netire network.
The convolutional and max pooling
layers are applied on the image until a fully connected layer is encountered,
then the image is flattened and processed as a 1D list of floats to predict
the fruit. The result will be a list of size n\_{fruits} and the index with the
maximum value will be the fruit the model predicted is in the image.

\begin{code}
feedForwardImage :: Image -> NeuralNetwork -> [Float]
feedForwardImage img (layer:layers) =
  case layer of
    ConvLayer conv       -> feedForwardImage (applyConvLayer img conv) layers
    MaxPoolingLayer size -> feedForwardImage (applyMaxPoolingLayer img size) layers
    -- Fully connected layer does not reconstruct an Image type for recursion
    -- but returns the final [Float] after computing all the fc layers using a foldl.
    FullyConnected _     -> feedForwardFullyConnected (flattenImage img) (layer:layers)
\end{code}

The \verb|randomKernel| function creates new convolution kernels for the convolutional layers to use.
Originaly, the kernels were initialized with completely random values, which could then be learned by also performing backwards propogation on the convolutional layers.
This would allow the model to train kernels to pick up on specific features in the images. However, implementing the backwards propogation for the convolutional layers was not succesful.
Now, the randomKernel function returns a gaussian convolutional kernel that is always initialized with the same distribution as to not introduce random noise into the input layer of the fully connected network,
However, this does introduce "blurring" of the input data. The CNN will therefore not perform nearly as well as a CNN that also trains its convolutional layers.
\begin{code}
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
  \includegraphics[width=0.9\linewidth]{assets/cnn.jpg}
  \caption{Example of the architecture of a Convolutional Neural Network.}
  \label{fig:cnn}
\end{figure}

The \verb|newModelCNN| function returns a newly initialized, untrained CNN. Figure \ref{fig:cnn} shows a visual example of the architecture of this network.
The features of the image are extracted in the convolutional and maxPooling layers, and the classification takes place in the fully connected layers.
This network also consits of two convolutional layers with maxPooling layers in between. The output of the last maxPooling layer is then flattened and will then be used as the input vector of the fully connected layers.
This model consists of two fully connected layers, with the first going from 8464 input neurons to 100 output neurons, and the second layer goes from 100 input neurons to n\_{fruittypes} output neurons, with one output neuron for each fruit type that the model can recognise.

\begin{code}
newModelCNN :: IO NeuralNetwork
newModelCNN = do
  -- First convolutional layer: 8 kernels (3x3)
  let conv1Kernels = replicateM 8 (randomKernel 3 0.1)
  conv1Biases  <- replicateM 8 (gauss 0.01)
  let convLayer1 = ConvLayer (conv1Kernels, conv1Biases)

  -- First max pooling layer with pool size 2x2
  let poolLayer1 = MaxPoolingLayer 2

  -- Second convolutional layer: 16 3x3 kernels
  let conv2Kernels = replicateM 16 (randomKernel 3 0.1)
  conv2Biases  <- replicateM 16 (gauss 0.01)
  let convLayer2 = ConvLayer (conv2Kernels, conv2Biases)

  -- Second max pooling layer with pool size 2x2
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

\end{code}

The \verb|newModelFC| function returns a newly initialized, untrained fully connected neural network without any convolutional or maxPooling layers.
\begin{code}
-- Model without convolutional layers
newModelFC :: IO NeuralNetwork
newModelFC = do
  -- Fully connected layer 1: 30000 -> 100
  fc1Biases  <- replicateM 100 (gauss 0.01)
  fc1Weights <- replicateM 100 (replicateM 30000 (gauss 0.01))
  let fcLayer1 = FullyConnected (fc1Biases, fc1Weights)

  -- Fully connected layer 1: 100 -> 2
  fc2Biases  <- replicateM 2 (gauss 0.01)
  fc2Weights <- replicateM 2 (replicateM 100 (gauss 0.01))
  let fcLayer2 = FullyConnected (fc2Biases, fc2Weights)

  return [fcLayer1, fcLayer2]
\end{code}

The \verb|forwardPass| function propogates an Image through the network to get the predicted probabilities of the network.
It recursively calls the feedForwards function for the convolutional layers and the max pooling layers as those functions have the same paramters.
For the fully connected layers, it calls the calculateFullyConnectedLayerOutput to get the final prediction of the fully connected layers.
\begin{code}
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
      -- Compute the delta for this layer by multiplying the propagated error with the derivative.
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
\end{code}

The \verb|replaceLayer| function updates a layer of the network with its new weights and biases by replacing it with the layer that has the updated values after the backwards propogation.
This function is used after the backpropogation function has calculated the new weights and biases of a layer in order to update the model.
\begin{code}
replaceLayer :: NeuralNetwork -> Layer -> Layer -> NeuralNetwork
replaceLayer [] _ _ = []
replaceLayer (l:ls) oldLayer newLayer
  | l == oldLayer = newLayer : ls
  | otherwise = l : replaceLayer ls oldLayer newLayer
\end{code}

The \verb|trainIteration| function performs a single training iteration.
First, the forwards pass is computed to get the prediction of the model on an input image.
Then the error is calculated between the prediction and the actual type of fruit in the image. This error is then propogated through the network using the \verb|backpropagateLayer| function to update the weights and biases of the fully connected layers.

\begin{code}
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

\end{code}

The \verb|trainModel| function trains the model for $n$ epochs by calling the trainIteration for every image in the training set.

\begin{code}
trainModel :: NeuralNetwork -> [(Image, [Float])] -> Int -> Float -> IO NeuralNetwork
trainModel initialModel trainingData epochs learningRate = do
  foldM trainEpoch initialModel [1..epochs]
  where
    trainEpoch model epoch = do
      putStrLn $ "Epoch " ++ show epoch ++ "/" ++ show epochs
      foldM (\currentModel img -> trainIteration currentModel img learningRate) model trainingData
\end{code}

\hide {
\begin{code}
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

predictFruit :: NeuralNetwork -> Image -> FruitType
predictFruit model image = toEnum (argmax (feedForwardImage image model))

-- Model Persistence
saveModel :: FilePath -> NeuralNetwork -> IO ()
saveModel filePath model = BL.writeFile filePath (encode model)

loadModel :: FilePath -> IO NeuralNetwork
loadModel filePath = decode <$> BL.readFile filePath
\end{code}
}