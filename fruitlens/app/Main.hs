{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BL

import Control.Monad
import Data.Functor
import Data.Ord
import Data.List
import System.Random


-- START SCOTTY API
import Web.Scotty (ActionM, scotty, post, jsonData, json, get, html, param)
import Data.Aeson 
import qualified Data.ByteString.Base64 as B64
import qualified Data.ByteString as BS 
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Base16 as B16
import Codec.Picture
import Data.Vector.Storable (toList)
import Data.List.Split (chunksOf)

newtype MyImage = MyImage { value :: String } deriving (Show)

instance FromJSON MyImage where
    parseJSON = withObject "Image" $ \o -> MyImage <$> o .: "image"

instance ToJSON MyImage where 
    toJSON (MyImage value) = object ["image" .= value]

main :: IO ()
main = scotty 8080 $ do
    post "/api/fruitlens" processImage
        
processImage :: ActionM ()
processImage = do
    (MyImage value) <- jsonData  
    let decoded = B64.decode (BC.pack value)
    case decoded of
        Left err -> Web.Scotty.json $ object ["error" .= ("Invalid Base64: " ++ err)] 
        Right byteArray -> do 
          case decodeImage byteArray of
                Left err -> Web.Scotty.json $ object ["error" .= ("Image decoding failed: " ++ err)]
                Right dynImage -> do
                    let converted = convertRGB8 dynImage
                    let rgbList = toList (imageData converted)
                    let tupleList = chunksOf 3 rgbList
                    let tupleList2d = chunksOf (imageWidth converted) tupleList
                    Web.Scotty.json $ object ["tuples" .= tupleList2d]

-- END SCOTTY API

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
  biases <- mapM (\n -> replicateM n (gauss 0.01)) outputLayers
  weights <- zipWithM (\m n -> replicateM n $ replicateM m $ gauss 0.01) layers outputLayers
  return (zip biases weights)

-- Activation function (ReLu)
activation :: Float -> Float
activation x | x > 0      = x
             | otherwise  = 0

-- Calculate the weighted sum of the inputs and the weights, and add the biases.
calculateLayerOutput :: [Float] -> Layer -> [Float]
calculateLayerOutput inputs (biases, weigths) = map activation $ zipWith (+) biases $ sum . zipWith (*) inputs <$> weigths

-- Calculate output of each layer in the neural network
feedForward :: [Float] -> NeuralNetwork -> [Float]
feedForward = foldl' calculateLayerOutput

-- main :: IO ()
-- main = do
--   initModel <- newModel [784, 30, 10]
--   let inputs = [1..784]
--   let outputs = feedForward inputs initModel
--   print outputs