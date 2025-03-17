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
import Codec.Picture.Png (encodePng)
import Data.Vector.Storable (toList)
import Data.List.Split (chunksOf)
import Codec.Picture.Extra (crop, scaleBilinear)
import Control.Monad.IO.Class (liftIO)

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
                    let square = toSquare converted --  Make it square before scaling to avoid distortion
                    let scaled = scaleBilinear 100 100 square -- Scale to 100 by 100 image
                    let rgbList = toList (imageData scaled) -- All RGB values in a row 
                    let tupleList = chunksOf 3 rgbList -- Convert to long list of [R, G, B] Tuples
                    let tupleList2d = chunksOf (imageWidth scaled) tupleList -- Convert to 2D list of [R, G, B] Tuples (top to bottom, left to right)
                    Web.Scotty.json $ object ["base64" .= BC.unpack (B64.encode (BL.toStrict (encodePng scaled))), "tuples" .= tupleList2d]
                    -- The scotty return above is only here to check the picture is being processed correctly, remove later
                    -- TODO: Send array to neural network

toSquare :: Image PixelRGB8 -> Image PixelRGB8 -- Crop image to square, centered
toSquare img =
    let w = imageWidth img
        h = imageHeight img
        size = min w h
        x = (w - size) `div` 2
        y = (h - size) `div` 2
    in crop x y size size img

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