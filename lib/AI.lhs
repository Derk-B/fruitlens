
The code for the neural network was taken from https://crypto.stanford.edu/~blynn/haskell/brain.html.
We adjusted it to make it compatible with learning types of fruits in images
\begin{code}

{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module AI where

import Control.Monad
import qualified Data.ByteString.Lazy as BS
import Data.List (foldl', maximumBy, transpose)
import Data.Ord
import qualified GHC.Int
import Utils (gauss)
import Data.Binary (encode, decode)
import qualified Data.ByteString.Lazy as BL
import System.Directory (doesFileExist)

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
newModel layers@(_ : outputLayers) = do
  let biases = map (`replicate` 1) outputLayers
  weights <- zipWithM (\m n -> replicateM n $ replicateM m (gauss 0.01)) layers outputLayers
  return (zip biases weights)

newBrain :: [Int] -> IO NeuralNetwork
newBrain szs@(_ : ts) =
  zip (flip replicate 1 <$> ts)
    <$> zipWithM (\m n -> replicateM n $ replicateM m $ gauss 0.01) szs ts

relu :: Float -> Float
relu = max 0

relu' :: (Ord a1, Num a1, Num a2) => a1 -> a2
relu' x
  | x < 0 = 0
  | otherwise = 1

calculateLayerOutput :: [Float] -> Layer -> [Float]
calculateLayerOutput layerInput (biases, weights) = zipWith (+) biases $ sum . zipWith (*) layerInput <$> weights

feedForward :: [Float] -> [([Float], [[Float]])] -> [Float]
feedForward = foldl' (((relu <$>) .) . calculateLayerOutput)

-- Returns a the activations and weighted inputs.
-- So in a layer we have a list of input values, we apply the weights to the inputs values and call it 'layerOutput'.
-- Then we apply the activation function (relu) to the 'layerOutput' and call it 'activatedOutput'
-- Then we return (activatedOutputs, layerOutputs)
getActivatedAndWeightedOutputs :: [Float] -> NeuralNetwork -> ([[Float]], [[Float]])
getActivatedAndWeightedOutputs initialInputs =
  foldl'
    ( \(inputs@(nodeInput : _), prevOutputs) layer ->
        let layerOutput = calculateLayerOutput nodeInput layer
         in ((relu <$> layerOutput) : inputs, layerOutput : prevOutputs)
    )
    ([initialInputs], [])

dCost :: (Num a, Ord a) => a -> a -> a
dCost a y
  | y == 1 && a >= y = 0
  | otherwise = a - y

deltas :: [Float] -> [Float] -> [([Float], [[Float]])] -> ([[Float]], [[Float]])
deltas initialInputs targets layers =
  let (activatedOutputs@(activatedValue : _), weightedOutput : wos) = getActivatedAndWeightedOutputs initialInputs layers
      -- Delta0 is the delta for the output layer.
      -- Backpropagation works in reverse, so that is why we first calculate the output layer delta.
      delta0 = zipWith (*) (zipWith dCost activatedValue targets) (relu' <$> weightedOutput)
   in (reverse activatedOutputs, f (transpose . snd <$> reverse layers) wos [delta0])
  where
    f _ [] dvs = dvs
    f (wm : wms) (zv : zvs) dvs@(dv : _) =
      f wms zvs $
        (: dvs) $
          zipWith (*) [sum $ zipWith (*) row dv | row <- wm] (relu' <$> zv)

learningRate :: Float
learningRate = 0.002

descend :: [Float] -> [Float] -> [Float]
descend av dv = zipWith (-) av ((learningRate *) <$> dv)

learn :: [Float] -> [Float] -> [([Float], [[Float]])] -> [([Float], [[Float]])]
learn inputs targets layers =
  let (avs, dvs) = deltas inputs targets layers
   in zip (zipWith descend (fst <$> layers) dvs) $
        zipWith3
          (\wvs av dv -> zipWith (\wv d -> descend wv ((d *) <$> av)) wvs dv)
          (snd <$> layers)
          avs
          dvs

getImage :: Num b => BL.ByteString -> GHC.Int.Int64 -> [b]
getImage s n = fromIntegral . BS.index s . (n * 28 ^ 2 + 16 +) <$> [0 .. 28 ^ 2 - 1]
getX :: Fractional b => BL.ByteString -> GHC.Int.Int64 -> [b]

getX s n = (/ 256) <$> getImage s n

getLabel :: Num b => BL.ByteString -> GHC.Int.Int64 -> b
getLabel s n = fromIntegral $ BS.index s (n + 8)

getY :: Num b => BL.ByteString -> GHC.Int.Int64 -> [b]
getY s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0 .. 9]

render :: (Integral a) => a -> Char
render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

train :: [[Float]] -> [[Float]] -> [[Float]] -> [[Float]] -> IO ()
train trainI trainL testI testL = do
  -- initialModel <- newBrain [30000, 512, 2] -- 243 / 1026

  -- Apple 6, apple 10, banana 1, banana 3, pear 1
  -- [30k, 20, 3] + 40 training items      => 700/1026
  -- [30k, 20, 3] + complete training data => 264/1026

  -- Apple 6, apple 10, banana 1, banana 3
  -- [30000, 32, 3] + 40 training items      => 826/826
  -- [30000, 32, 2] + complete training data => 802/826
  -- Initialize the model
  let modelFile = "trained_model.bin"

  -- Check if the model file exists
  modelExists <- doesFileExist modelFile

  if modelExists
    then do
      -- Load the model from the file
      smartModel <- loadModel modelFile
      putStrLn "Model loaded from trained_model.bin"

      -- Test the loaded model
      let bestOf = fst . maximumBy (comparing snd) . zip ([0 ..] :: [Float])
      let guesses = bestOf . (`feedForward` smartModel) <$> testI
      let answers = bestOf <$> testL
      putStrLn $ show (sum $ fromEnum <$> zipWith (==) guesses answers) ++ " / " ++ show (length testL)

      print (head guesses, head answers, head testL)
    else do
      -- Train the model
      initialModel <- newBrain [30000, 32, 2]
      let smartModel = foldl' (\net (input, label) -> learn input label net) initialModel $ zip trainI trainL

      -- Save the trained model to a file
      saveModel modelFile smartModel
      putStrLn "Model saved to trained_model.bin"

      -- Test the model
      let bestOf = fst . maximumBy (comparing snd) . zip ([0 ..] :: [Float])
      let guesses = bestOf . (`feedForward` smartModel) <$> testI
      let answers = bestOf <$> testL
      putStrLn $ show (sum $ fromEnum <$> zipWith (==) guesses answers) ++ " / " ++ show (length testL)

      print (head guesses, head answers, head testL)
  return ()

-- | Save the neural network model to a file
saveModel :: FilePath -> NeuralNetwork -> IO ()
saveModel filePath model = BL.writeFile filePath (encode model)

-- | Load the neural network model from a file
loadModel :: FilePath -> IO NeuralNetwork
loadModel filePath = decode <$> BL.readFile filePath

\end{code}