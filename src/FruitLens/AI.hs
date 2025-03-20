{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}
{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
module FruitLens.AI where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS

import Control.Monad
import Data.Ord
import Data.List ( foldl', maximumBy, transpose )
import System.Random

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

relu :: Float -> Float
relu = max 0

relu' :: (Ord a1, Num a1, Num a2) => a1 -> a2
relu' x | x < 0      = 0
        | otherwise  = 1

calculateLayerOutput :: [Float] -> Layer -> [Float]
calculateLayerOutput layerInput (biases, weights) = zipWith (+) biases $ sum . zipWith (*) layerInput <$> weights

feedForward :: [Float] -> [([Float], [[Float]])] -> [Float]
feedForward = foldl' (((relu <$>) . ) . calculateLayerOutput)

-- Returns a the activations and weighted inputs.
-- So in a layer we have a list of input values, we apply the weights to the inputs values and call it 'layerOutput'.
-- Then we apply the activation function (relu) to the 'layerOutput' and call it 'activatedOutput'
-- Then we return (activatedOutputs, layerOutputs)
getActivatedAndWeightedOutputs :: [Float] -> NeuralNetwork -> ([[Float]], [[Float]])
getActivatedAndWeightedOutputs initialInputs = foldl' (\(inputs@(nodeInput:_), prevOutputs) layer -> 
  let layerOutput = calculateLayerOutput nodeInput layer 
  in ((relu <$> layerOutput):inputs, layerOutput:prevOutputs)) ([initialInputs], [])

dCost :: (Num a, Ord a) => a -> a -> a
dCost a y | y == 1 && a >= y = 0
          | otherwise        = a - y

deltas :: [Float] -> [Float] -> [([Float], [[Float]])] -> ([[Float]], [[Float]])
deltas initialInputs targets layers = let
  (activatedOutputs@(activatedValue:_), weightedOutput:wos) = getActivatedAndWeightedOutputs initialInputs layers
  -- Delta0 is the delta for the output layer. 
  -- Backpropagation works in reverse, so that is why we first calculate the output layer delta.
  delta0 = zipWith (*) (zipWith dCost activatedValue targets) (relu' <$> weightedOutput)
  in (reverse activatedOutputs, f (transpose . snd <$> reverse layers) wos [delta0]) where
    f _ [] dvs = dvs
    f (wm:wms) (zv:zvs) dvs@(dv:_) = f wms zvs $ (:dvs) $
      zipWith (*) [sum $ zipWith (*) row dv | row <- wm] (relu' <$> zv)

learningRate :: Float
learningRate = 0.002

descend :: [Float] -> [Float] -> [Float]
descend av dv = zipWith (-) av ((learningRate *) <$> dv)

learn :: [Float] -> [Float] -> [([Float], [[Float]])] -> [([Float], [[Float]])]
learn inputs targets layers = let (avs, dvs) = deltas inputs targets layers
  in zip (zipWith descend (fst <$> layers) dvs) $
    zipWith3 (\wvs av dv -> zipWith (\wv d -> descend wv ((d*) <$> av)) wvs dv)
      (snd <$> layers) avs dvs

getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]
getX     s n = (/ 256) <$> getImage s n
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY     s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]

render :: Integral a => a -> Char
render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

main :: IO ()
main = do
  [trainI, trainL, testI, testL] <- mapM ((decompress  <$>) . BS.readFile)
    [ "train-images-idx3-ubyte.gz"
    , "train-labels-idx1-ubyte.gz"
    ,  "t10k-images-idx3-ubyte.gz"
    ,  "t10k-labels-idx1-ubyte.gz"
    ]
  b <- newModel [784, 30, 10]
  n <- (`mod` 10000) <$> randomIO
  putStr . unlines $
    take 28 $ take 28 <$> iterate (drop 28) (render <$> getImage testI n)

  let
    example = getX testI n
    bs = scanl (foldl' (\b n -> learn (getX trainI n) (getY trainL n) b)) b [
     [   0.. 999],
     [1000..2999],
     [3000..5999],
     [6000..9999]]
    smart = last bs
    cute d score = show d ++ ": " ++ replicate (round $ 70 * min 1 score) '+'
    bestOf = fst . maximumBy (comparing snd) . zip [0..]

  forM_ bs $ putStrLn . unlines . zipWith cute [0..9] . feedForward example

  putStrLn $ "best guess: " ++ show (bestOf $ feedForward example smart)

  let guesses = bestOf . (\n -> feedForward (getX testI n) smart) <$> [0..9999]
  let answers = getLabel testL <$> [0..9999]
  putStrLn $ show (sum $ fromEnum <$> zipWith (==) guesses answers) ++
    " / 10000"